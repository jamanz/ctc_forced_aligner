import os
import logging
import numpy

# ONNX specific
import onnxruntime
from .onnx_utils import generate_emissions_onnx
from .constants import DEFAULT_ONNX_MODEL_URL, DEFAULT_MODEL_PATH, VOCAB_DICT, SAMPLING_FREQ
from .tokenizer_wrapper import Tokenizer as OnnxTokenizer  # Alias for clarity
from .text_utils import preprocess_transcript_for_alignment
from .utils import (
    ensure_model_downloaded, Segment, merge_repeats, get_spans,
    load_audio as util_load_audio  # Alias to avoid confusion
)
from .ctc_aligner import align_sequences as ctc_align_cpp  # C++ aligner
from .subtitles import (
    _postprocess_word_level_alignments, generate_srt_content, generate_webvtt_content
)

# PyTorch specific
import torch
import torchaudio  # Keep for pipeline definitions
from .torch_utils import get_word_stamps_pytorch_chunked
from .constants import (
    DEFAULT_CHUNK_LENGTH_SEC,
    DEFAULT_OVERLAP_LENGTH_SEC,
    DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS
)

logger = logging.getLogger(__name__)


# --- ONNX Alignment Classes ---
class Alignment:
    def __init__(self, onnx_model_path: str = DEFAULT_MODEL_PATH, model_url: str = DEFAULT_ONNX_MODEL_URL):
        ensure_model_downloaded(onnx_model_path, model_url)
        logger.info(f"Loading ONNX alignment model from '{onnx_model_path}'...")
        self.session = onnxruntime.InferenceSession(onnx_model_path,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Check if CUDA is available and used
        if 'CUDAExecutionProvider' in self.session.get_providers():
            logger.info("ONNX Runtime is using CUDAExecutionProvider.")
        else:
            logger.info("ONNX Runtime is using CPUExecutionProvider.")

        self.tokenizer = OnnxTokenizer(VOCAB_DICT)  # Uses the package's default vocab
        self._cached_align_data = {}  # For caching results of align_audio_transcript

    def _get_onnx_alignments(
            self,
            emissions: numpy.ndarray,  # Log probs
            tokens_for_model: list,
            # List of space-separated chars, e.g. ["<star>", "h e l l o", "<star>", "w o r l d"]
    ):
        # This function adapts the original get_alignments logic for ONNX
        if not tokens_for_model:
            raise ValueError("Empty token list provided for alignment.")

        # Vocabulary including <star> for ONNX model
        # The ONNX model from original repo expects <star>
        onnx_vocab_with_star = self.tokenizer.get_vocab().copy()
        if "<star>" not in onnx_vocab_with_star:
            onnx_vocab_with_star["<star>"] = len(onnx_vocab_with_star)

        token_indices_list = []
        for token_str in tokens_for_model:  # Each item in tokens_for_model is already a "word" or <star>
            if token_str == "<star>":
                token_indices_list.append(onnx_vocab_with_star["<star>"])
            else:  # It's a sequence of space-separated characters
                chars = token_str.split(" ")
                for char_token in chars:
                    if char_token in onnx_vocab_with_star:
                        token_indices_list.append(onnx_vocab_with_star[char_token])
                    else:
                        logger.warning(f"ONNX: Token '{char_token}' not found in vocabulary. Using UNK.")
                        token_indices_list.append(self.tokenizer.unk_token_id)

        if not token_indices_list:
            raise ValueError("Failed to convert any tokens to indices for ONNX alignment.")

        targets = numpy.asarray([token_indices_list], dtype=numpy.int64)
        blank_id = self.tokenizer.blank_token_id

        # Validate targets against emissions shape and blank_id
        if blank_id in targets:
            raise ValueError(f"ONNX targets array should not contain blank index ({blank_id}).")
        if blank_id >= emissions.shape[-1] or blank_id < 0:
            raise ValueError(f"ONNX blank_id {blank_id} must be within [0, {emissions.shape[-1]})")
        if numpy.max(targets) >= emissions.shape[-1] or numpy.min(targets) < 0:
            raise ValueError(
                f"ONNX targets values must be within [0, {emissions.shape[-1]}). Max target: {numpy.max(targets)}")

        # Ensure emissions are in the shape (1, T, C) for ctc_align_cpp
        if emissions.ndim == 2:
            emissions_for_align = numpy.expand_dims(emissions, axis=0)
        elif emissions.ndim == 3 and emissions.shape[0] == 1:
            emissions_for_align = emissions
        else:
            raise ValueError(f"Unexpected emissions shape for ONNX alignment: {emissions.shape}")

        paths_cpp, scores_cpp = ctc_align_cpp(emissions_for_align, targets, blank_id)

        path_list = paths_cpp.squeeze().tolist()

        idx_to_token_map = {v: k for k, v in onnx_vocab_with_star.items()}
        ctc_segments = merge_repeats(path_list, idx_to_token_map)  # List of Segments (char-level)

        return ctc_segments, scores_cpp.squeeze(), idx_to_token_map[blank_id]

    def align_audio_transcript(
            self,
            audio_path: str,
            transcript_text: str,  # Raw transcript text
            language: str = "eng",
            onnx_batch_size: int = 4,  # Batch size for ONNX window processing
            romanize_for_onnx: bool = True,  # ONNX model usually expects romanized, char-spaced
            star_insertion_strategy: str = "segment"  # "segment", "edges", "none"
    ) -> Dict[str, Any]:
        """
        Core alignment logic for ONNX.
        Returns a dictionary with word_timestamps, original_lines, and other details.
        """
        # Check cache
        cache_key = (audio_path, transcript_text, language, onnx_batch_size, romanize_for_onnx, star_insertion_strategy)
        if cache_key in self._cached_align_data:
            logger.debug("Returning cached ONNX alignment data.")
            return self_cached_align_data[cache_key]

        audio_waveform = util_load_audio(audio_path, target_sr=SAMPLING_FREQ)

        # Preprocess transcript: normalize, (optional) romanize, add <star> tokens
        # tokens_for_model: list of space-separated characters per "word" or <star>
        # text_units_starred: list of original words/segments with <star> tokens
        tokens_for_model, text_units_starred = preprocess_transcript_for_alignment(
            transcript_text,
            language=language,
            romanize=romanize_for_onnx,
            split_unit="word",  # ONNX aligner typically works well with word-level segments for text_units_starred
            star_insertion_strategy=star_insertion_strategy
        )

        if not tokens_for_model or not text_units_starred:
            logger.warning("Preprocessing transcript resulted in empty tokens/text units. Cannot align.")
            return {"word_timestamps": [], "original_lines": transcript_text.splitlines(),
                    "error": "Empty preprocessed transcript"}

        emissions, stride_sec = generate_emissions_onnx(
            self.session, audio_waveform, batch_size=onnx_batch_size
        )
        if emissions.size == 0:
            logger.error("ONNX emission generation failed or produced empty output.")
            return {"word_timestamps": [], "original_lines": transcript_text.splitlines(),
                    "error": "Empty emissions from ONNX model"}

        ctc_segments, frame_scores, blank_token_str = self._get_onnx_alignments(emissions, tokens_for_model)

        # `get_spans` maps these character segments back to the `text_units_starred`
        # It expects `text_units_starred` to be a list of "words" or "<star>"
        # and `tokens_for_model` to be the char-representation of those words.
        # The current `preprocess_transcript_for_alignment` returns `tokens_for_model` as list of char-strings.
        # We need to ensure `get_spans` can link `ctc_segments` (char level)
        # to `text_units_starred` (word/segment level) using `tokens_for_model` as the bridge.
        # The `get_spans` in utils.py needs to be robust for this.
        # Let's pass `tokens_for_model` to `get_spans` as the "tokens to map" argument.

        # The original `get_spans` took `tokens_starred` (which were like text_units_starred).
        # If tokens_for_model is ["<star>", "h e l l o", "<star>", "w o r l d"]
        # and text_units_starred is ["<star>", "Hello", "<star>", "World"]
        # then `get_spans` should map ctc_segments to `text_units_starred` based on the structure of `tokens_for_model`.

        # The `get_spans` function needs the character sequence corresponding to text_units_starred.
        # `tokens_for_model` is exactly that: a list where each element is the char sequence for an element in `text_units_starred`.
        # e.g. text_units_starred = ["<star>", "Hi"]
        #      tokens_for_model = ["<star>", "h i"]
        spans_per_text_unit = get_spans(tokens_for_model, ctc_segments, blank_token_str)

        # Filter out "<star>" units before postprocessing for timestamps
        # We need to align `spans_per_text_unit` with `text_units_starred` and then filter.

        valid_text_units = []
        valid_spans = []
        for i, unit_text in enumerate(text_units_starred):
            if unit_text != "<star>":
                valid_text_units.append(unit_text)
                if i < len(spans_per_text_unit):
                    valid_spans.append(spans_per_text_unit[i])
                else:  # Should not happen if lengths match
                    valid_spans.append([Segment(blank_token_str, 0, -1)])  # Placeholder for safety

        word_timestamps = _postprocess_word_level_alignments(
            text_units=valid_text_units,
            word_spans=valid_spans,
            total_emission_frames=emissions.shape[0],
            model_output_stride_sec=stride_sec,
            alignment_scores_per_frame=torch.from_numpy(frame_scores)  # Convert to tensor for type hint
        )

        original_lines = [line for line in transcript_text.splitlines() if line.strip()]

        result = {"word_timestamps": word_timestamps, "original_lines": original_lines}
        self._cached_align_data[cache_key] = result
        return result

    def generate_srt(self, audio_path: str, transcript_text: str, language: str = "eng", **kwargs) -> str:
        align_data = self.align_audio_transcript(audio_path, transcript_text, language, **kwargs)
        return generate_srt_content(align_data["word_timestamps"], align_data["original_lines"])

    def generate_webvtt(self, audio_path: str, transcript_text: str, language: str = "eng", **kwargs) -> str:
        align_data = self.align_audio_transcript(audio_path, transcript_text, language, **kwargs)
        return generate_webvtt_content(align_data["word_timestamps"], align_data["original_lines"])


class AlignmentSingleton(Alignment):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            model_path = kwargs.get("onnx_model_path", DEFAULT_MODEL_PATH)
            model_url = kwargs.get("model_url", DEFAULT_ONNX_MODEL_URL)
            cls._instance = super(AlignmentSingleton, cls).__new__(cls)
            # Explicitly call __init__ of the superclass (Alignment)
            # This is not standard for __new__ but needed here because Alignment.__init__ does the loading.
            # A better pattern might be to have a separate _load method.
            super(AlignmentSingleton, cls._instance).__init__(onnx_model_path=model_path, model_url=model_url)
        return cls._instance

    # No need to override methods if they are the same as Alignment


# --- PyTorch Alignment Classes ---
class AlignmentTorch:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch Aligner initialized on device: {self.device}")
        self.loaded_model_instance = None
        self.loaded_model_bundle_name = None  # To track which bundle is loaded
        self._cached_align_data = {}

    def _load_pytorch_model(self, model_type_name: str):
        if self.loaded_model_instance is not None and self.loaded_model_bundle_name == model_type_name:
            return self.loaded_model_instance  # Return cached model

        logger.info(f"Loading PyTorch model bundle: {model_type_name}...")
        try:
            if model_type_name == 'MMS_FA':  # Default, specific handling
                bundle = torchaudio.pipelines.MMS_FA
            elif hasattr(torchaudio.pipelines, model_type_name):
                bundle = getattr(torchaudio.pipelines, model_type_name)
            else:
                raise ValueError(f"Unsupported PyTorch model type: {model_type_name}")

            self.loaded_model_instance = bundle.get_model().to(self.device)
            self.loaded_model_bundle_name = model_type_name
            self.current_bundle = bundle  # Store bundle for other info like labels, sample_rate
            logger.info(f"PyTorch model {model_type_name} loaded successfully on {self.device}.")
            return self.loaded_model_instance
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {model_type_name}: {e}")
            raise

    def align_audio_transcript(
            self,
            audio_path: str,
            transcript_path_or_text: str,  # Path to transcript file OR raw text
            model_type: str = 'MMS_FA',  # Default MMS_FA, or e.g., 'WAV2VEC2_ASR_BASE_960H'
            chunk_length_sec: float = DEFAULT_CHUNK_LENGTH_SEC,
            overlap_length_sec: float = DEFAULT_OVERLAP_LENGTH_SEC,
            batch_size_chunks: int = DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS,
            language: str = "eng"  # For text normalization if transcript is raw text
    ) -> Dict[str, Any]:
        """
        Core alignment logic for PyTorch models using chunking.
        Returns a dictionary with word_timestamps, original_lines.
        """
        cache_key = (audio_path, transcript_path_or_text, model_type, chunk_length_sec, overlap_length_sec,
                     batch_size_chunks, language)
        if cache_key in self._cached_align_data:
            logger.debug("Returning cached PyTorch alignment data.")
            return self._cached_align_data[cache_key]

        model_instance = self._load_pytorch_model(model_type)

        # Call the chunked processing function
        (
            transcript_words, word_char_spans, total_emission_frames,
            alignment_scores_frames, model_output_stride_sec,
            _, processed_lyrics_lines  # model instance is already self.loaded_model_instance
        ) = get_word_stamps_pytorch_chunked(
            audio_path=audio_path,
            transcript_path_or_text=transcript_path_or_text,
            model_bundle=self.current_bundle,  # Pass the current bundle
            device=self.device,
            model_instance=model_instance,
            chunk_length_sec=chunk_length_sec,
            overlap_length_sec=overlap_length_sec,
            batch_size_chunks=batch_size_chunks,
            language_for_normalization=language
        )

        # Now, call the postprocessing function
        word_timestamps = _postprocess_word_level_alignments(
            text_units=transcript_words,
            word_spans=word_char_spans,
            total_emission_frames=total_emission_frames,
            model_output_stride_sec=model_output_stride_sec,
            alignment_scores_per_frame=alignment_scores_frames
        )

        result = {"word_timestamps": word_timestamps, "original_lines": processed_lyrics_lines}
        self._cached_align_data[cache_key] = result
        return result

    def generate_srt(self, audio_path: str, transcript_path_or_text: str, model_type: str = 'MMS_FA', **kwargs) -> str:
        language = kwargs.pop("language", "eng")  # Extract language if provided for align_audio_transcript
        align_data = self.align_audio_transcript(audio_path, transcript_path_or_text, model_type, language=language,
                                                 **kwargs)
        return generate_srt_content(align_data["word_timestamps"], align_data["original_lines"])

    def generate_webvtt(self, audio_path: str, transcript_path_or_text: str, model_type: str = 'MMS_FA',
                        **kwargs) -> str:
        language = kwargs.pop("language", "eng")
        align_data = self.align_audio_transcript(audio_path, transcript_path_or_text, model_type, language=language,
                                                 **kwargs)
        return generate_webvtt_content(align_data["word_timestamps"], align_data["original_lines"])


class AlignmentTorchSingleton(AlignmentTorch):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AlignmentTorchSingleton, cls).__new__(cls)
            # Call __init__ of the superclass (AlignmentTorch)
            super(AlignmentTorchSingleton, cls._instance).__init__()
        return cls._instance

    # Methods are inherited from AlignmentTorch