import os

import torch
import torchaudio
import torchaudio.functional as F
import numpy
import math
import logging
from typing import List, Tuple, Dict, Optional, Any

from .constants import (
    SAMPLING_FREQ,
    DEFAULT_CHUNK_LENGTH_SEC,
    DEFAULT_OVERLAP_LENGTH_SEC,
    DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS
)
from .utils import load_audio as util_load_audio, Segment, unflatten  # To avoid confusion with torch load
from .text_utils import load_transcript_from_file, text_normalize

logger = logging.getLogger(__name__)


def _get_model_output_stride_sec(model, audio_input_length_samples: int, sample_rate: int, device) -> float:
    """
    Estimates the model's output stride (seconds per frame) by making a dummy pass.
    This is crucial for correctly handling overlaps in chunked processing.
    """
    if audio_input_length_samples == 0:
        # Default for Wav2Vec2-like models: 20ms frames
        return 0.02

    try:
        dummy_input = torch.randn(1, audio_input_length_samples, device=device, dtype=torch.float32)
        with torch.inference_mode():
            model.eval()  # Ensure model is in eval mode
            emissions_dummy, output_lengths_dummy = model(dummy_input)
            # emissions_dummy shape: (batch, time_frames, num_classes)
            # output_lengths_dummy might be None or specify actual frames if input has padding

            num_output_frames = emissions_dummy.shape[1]
            if num_output_frames == 0:
                logger.warning(
                    "Dummy pass for stride calculation resulted in 0 output frames. Defaulting stride to 20ms.")
                return 0.02  # Default if no frames (e.g. input too short for model's receptive field)

            audio_duration_sec = audio_input_length_samples / sample_rate
            stride_sec = audio_duration_sec / num_output_frames
            logger.info(f"Estimated model output stride: {stride_sec:.6f} seconds/frame "
                        f"({(1 / stride_sec if stride_sec > 0 else 0):.2f} Hz output frame rate)")
            return stride_sec
    except Exception as e:
        logger.warning(f"Error during dummy pass for stride calculation: {e}. Defaulting stride to 20ms.")
        return 0.02  # Default for Wav2Vec2-like models: 20ms frames


def _create_audio_chunks_pytorch(
        waveform_tensor: torch.Tensor,  # Shape (1, num_samples) or (num_samples)
        sample_rate: int,
        chunk_length_sec: float,
        overlap_length_sec: float
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Slices a waveform tensor into overlapping chunks for PyTorch model processing.
    Returns a list of chunk tensors and a list of their effective (unpadded) lengths.
    """
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # Ensure (1, num_samples)

    num_total_samples = waveform_tensor.shape[1]

    chunk_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_length_sec * sample_rate)

    # Each chunk fed to the model will have overlap on both sides for context,
    # except for the first and last chunks of the audio.
    # model_input_chunk_len = chunk_samples + 2 * overlap_samples (for internal chunks)
    # For simplicity here, let's define chunk_for_model as the segment including its *right* overlap
    # and step by the non-overlapping part.

    # A common strategy: define model_process_len and step_len
    # model_process_len: total samples fed to model for one chunk (e.g., 20s chunk + 2s left_ctx + 2s right_ctx = 24s)
    # useful_chunk_len: the part of the chunk whose output we keep (e.g., 20s)
    # step_len: how much we advance for the next useful_chunk_len (e.g., 20s)

    # Let's use the simpler definition:
    # - chunk_len_model_input: what we feed to the model (e.g. 20s + 2s_right_overlap)
    # - step: main part of the chunk (e.g. 20s)
    # This means left overlap needs to be handled by taking previous chunk's end.

    # Alternative: each model input is chunk_len + left_ctx + right_ctx
    # Let's adopt the context_window approach: each model input is `chunk_length_sec` plus context on both sides.
    # The "useful" part corresponds to `chunk_length_sec`.

    model_feed_len = chunk_samples + 2 * overlap_samples  # Total samples for model per chunk
    step_len = chunk_samples  # Advance by the main chunk duration

    chunks = []
    effective_lengths = []  # Original length of audio in each chunk before padding

    current_pos = 0
    while current_pos < num_total_samples:
        # Define the segment to extract, including potential context
        # Context for the current chunk_samples starting at current_pos
        actual_chunk_start_in_waveform = current_pos
        actual_chunk_end_in_waveform = current_pos + chunk_samples

        # Determine context window for this chunk
        context_start = max(0, actual_chunk_start_in_waveform - overlap_samples)
        context_end = min(num_total_samples, actual_chunk_end_in_waveform + overlap_samples)

        # Extract the slice for model input
        model_input_slice = waveform_tensor[:, context_start:context_end]
        current_effective_length = model_input_slice.shape[1]

        # Pad if this slice is shorter than model_feed_len (e.g. at audio start/end)
        if current_effective_length < model_feed_len:
            padding_needed = model_feed_len - current_effective_length
            # Pad on the right. For first chunk, left might also be "padded" by shorter context.
            model_input_slice = torch.nn.functional.pad(model_input_slice, (0, padding_needed), mode='constant',
                                                        value=0.0)

        chunks.append(model_input_slice)
        effective_lengths.append(current_effective_length)  # Store original length before padding to model_feed_len

        current_pos += step_len
        if step_len == 0:  # Safety break for misconfiguration
            logger.error("Step length is zero, breaking chunk creation loop.")
            break

    if not chunks and num_total_samples > 0:  # Audio is shorter than one chunk_samples effectively
        context_start = 0
        context_end = num_total_samples
        model_input_slice = waveform_tensor[:, context_start:context_end]
        current_effective_length = model_input_slice.shape[1]
        if current_effective_length < model_feed_len:
            padding_needed = model_feed_len - current_effective_length
            model_input_slice = torch.nn.functional.pad(model_input_slice, (0, padding_needed), mode='constant',
                                                        value=0.0)
        chunks.append(model_input_slice)
        effective_lengths.append(current_effective_length)

    return chunks, effective_lengths


def get_word_stamps_pytorch_chunked(
        audio_path: str,
        transcript_path_or_text: str,  # Can be file path or direct text
        model_bundle: Any,  # e.g., torchaudio.pipelines.MMS_FA
        device: torch.device,
        model_instance: Optional[torch.nn.Module] = None,  # Allow passing loaded model
        chunk_length_sec: float = DEFAULT_CHUNK_LENGTH_SEC,
        overlap_length_sec: float = DEFAULT_OVERLAP_LENGTH_SEC,
        batch_size_chunks: int = DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS,
        language_for_normalization: str = "eng"  # For text_normalize if transcript is raw
) -> Tuple[List[Dict[str, Any]], torch.nn.Module, List[str]]:
    """
    Performs forced alignment using PyTorch models from torchaudio.pipelines,
    with audio processed in chunks to handle long files and reduce memory.
    """
    waveform_np = util_load_audio(audio_path, target_sr=model_bundle.sample_rate)
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(device)  # (1, num_samples)

    if model_instance is None:
        logger.info(f"Loading PyTorch model from bundle: {model_bundle}")
        model = model_bundle.get_model().to(device)
    else:
        model = model_instance
    model.eval()

    # --- Determine model's output stride (seconds per emission frame) ---
    # This is crucial for knowing how many emission frames correspond to overlap_samples
    # Try a short dummy input if audio is long enough, else use a default
    stride_estimation_len_samples = min(waveform_tensor.shape[1],
                                        model_bundle.sample_rate * 5)  # Use up to 5s for estimation
    model_output_stride_sec = _get_model_output_stride_sec(model, stride_estimation_len_samples,
                                                           model_bundle.sample_rate, device)

    if model_output_stride_sec <= 0:
        raise ValueError("Could not determine a valid model output stride.")

    overlap_frames_per_side = math.ceil(overlap_length_sec / model_output_stride_sec)
    logger.info(
        f"Chunking: chunk_len={chunk_length_sec}s, overlap={overlap_length_sec}s ({overlap_frames_per_side} frames/side), batch_size_chunks={batch_size_chunks}")

    # --- Prepare Chunks ---
    audio_chunks, chunk_effective_lengths = _create_audio_chunks_pytorch(
        waveform_tensor, model_bundle.sample_rate, chunk_length_sec, overlap_length_sec
    )

    if not audio_chunks:
        logger.warning(f"No audio chunks created for {audio_path}. Returning empty results.")
        return [], model, []

    # --- Process Chunks and Stitch Emissions ---
    all_stitched_emissions = []
    num_processed_chunks = 0

    for i in range(0, len(audio_chunks), batch_size_chunks):
        batch_chunk_list = audio_chunks[i: i + batch_size_chunks]
        batch_lengths_list = chunk_effective_lengths[i: i + batch_size_chunks]  # Original lengths of audio in chunks

        # Stack chunks into a batch. All chunks from _create_audio_chunks_pytorch are already padded to same model_feed_len
        batch_tensor = torch.cat(batch_chunk_list, dim=0).to(device)  # (batch_size_chunks, num_samples_model_feed)

        # Create attention_mask or lengths_tensor if needed by the model
        # For torchaudio.pipelines models, they often handle this internally or don't require explicit mask
        # if inputs are fixed length. If variable length inputs were batched *before* padding to model_feed_len,
        # then a lengths tensor would be needed for torchaudio.models.Wav2Vec2Model.
        # Here, all inputs to model in batch_tensor are of model_feed_len.
        # The `output_lengths` from model will give actual emission frames.

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits_batch, output_lengths_batch = model(batch_tensor)  # output_lengths_batch is for emissions
            log_probs_batch = F.log_softmax(logits_batch, dim=-1)  # (batch_size_chunks, time_frames, num_classes)

        # Process each chunk's emissions from the batch
        for j in range(log_probs_batch.shape[0]):
            chunk_log_probs = log_probs_batch[j, :output_lengths_batch[j], :]  # Use actual output length

            # Determine how much to trim based on current chunk's position
            is_first_chunk_of_audio = (num_processed_chunks == 0)
            is_last_chunk_of_audio = (num_processed_chunks == len(audio_chunks) - 1)

            trim_start_frames = 0 if is_first_chunk_of_audio else overlap_frames_per_side
            trim_end_frames = chunk_log_probs.shape[0] if is_last_chunk_of_audio else chunk_log_probs.shape[
                                                                                          0] - overlap_frames_per_side

            # Ensure trim indices are valid
            trim_start_frames = min(trim_start_frames, chunk_log_probs.shape[0])
            trim_end_frames = max(trim_start_frames, min(trim_end_frames, chunk_log_probs.shape[0]))

            if trim_start_frames < trim_end_frames:
                stitched_part = chunk_log_probs[trim_start_frames:trim_end_frames, :]
                all_stitched_emissions.append(stitched_part.cpu())  # Move to CPU to free GPU VRAM
            elif chunk_log_probs.shape[0] > 0 and (
                    is_first_chunk_of_audio and is_last_chunk_of_audio):  # Single chunk audio
                all_stitched_emissions.append(chunk_log_probs.cpu())

            num_processed_chunks += 1

    if not all_stitched_emissions:
        logger.warning(f"No emissions were generated after chunk processing for {audio_path}. Returning empty results.")
        # Create a dummy tiny emission to prevent downstream errors if expected
        dummy_vocab_size = model_bundle.get_dict().__len__() if hasattr(model_bundle, 'get_dict') else 32
        final_emissions = torch.empty(1, 0, dummy_vocab_size, dtype=torch.float32)  # (B=1, T=0, C)
    else:
        final_emissions = torch.cat(all_stitched_emissions, dim=0).unsqueeze(0)  # (1, TotalFrames, NumClasses)

    logger.info(f"Final stitched emissions shape: {final_emissions.shape}")

    # --- Prepare Transcript ---
    # For torchaudio pipelines (like MMS_FA), the dictionary often includes special tokens
    # and expects characters, sometimes space-separated or with special markers like '|'.
    # The `load_transcript_from_file` is a bit generic.
    # It's better to use the bundle's dictionary and tokenize according to its expectations.

    labels = model_bundle.get_labels()  # Usually a tuple or list of characters/tokens
    # For char-based models, this might be simple like ('<s>', '<pad>', ..., '|', 'e', 't', 'a', ...)
    # For BPE/WordPiece models, it would be subword units. MMS_FA is char-based.

    # Create a dictionary mapping from char to id
    # This is a simplified version; proper tokenization depends on the model.
    # Many torchaudio pipelines expect character sequences, possibly with spaces represented by a specific token.
    # For MMS_FA, the dictionary from `get_dict()` is `{'<s>': 0, ..., '|': 4, ...}` where '|' is word boundary.

    char_to_id = {char: i for i, char in enumerate(labels)}  # This is a simplified dictionary.
    # For MMS_FA, bundle.get_dict() is more accurate.

    # Load and prepare transcript
    if os.path.exists(transcript_path_or_text):
        with open(transcript_path_or_text, "r", encoding="utf-8") as f:
            raw_text_lines = [line.strip() for line in f if line.strip()]
            transcript_text_for_processing = " ".join(raw_text_lines)
    else:  # Input is direct text
        raw_text_lines = transcript_path_or_text.split('\n')
        raw_text_lines = [line.strip() for line in raw_text_lines if line.strip()]
        transcript_text_for_processing = transcript_path_or_text.replace("\n", " ").strip()

    # Normalize text for tokenization (e.g. lowercase, remove certain punctuation)
    # This step should align with how the model was trained.
    # For MMS_FA, it's generally lowercase characters.
    normalized_transcript_text = text_normalize(transcript_text_for_processing, iso_code=language_for_normalization)

    # Tokenize: For char-based models like MMS_FA, this means getting char IDs.
    # Words should be separated by the word boundary token if the model uses one (e.g., '|' for MMS).
    # The MMS_FA bundle's `get_dict` often includes '|' as a token representing space or word boundary.

    tokenized_transcript_chars = []
    # Split normalized text into words to insert word boundary tokens
    words = normalized_transcript_text.split()
    word_boundary_token = '|'  # Common for MMS models

    for i, word in enumerate(words):
        for char_token in list(word):
            if char_token in char_to_id:
                tokenized_transcript_chars.append(char_to_id[char_token])
            else:
                logger.warning(f"Character '{char_token}' not in model labels. Skipping.")
        if i < len(words) - 1:  # Add word boundary token between words
            if word_boundary_token in char_to_id:
                tokenized_transcript_chars.append(char_to_id[word_boundary_token])
            else:  # If no explicit word boundary, can try space if in vocab, or just concat chars
                if ' ' in char_to_id: tokenized_transcript_chars.append(char_to_id[' '])

    if not tokenized_transcript_chars:
        logger.warning("Empty tokenized transcript. Alignment will likely fail or be empty.")
        # Handle empty transcript: create a dummy target or return empty
        # For forced_align, an empty target might lead to errors.
        # Let's make it a sequence with just a pad or blank if that helps, or ensure it's non-empty.
        # If targets must be non-empty, perhaps a single blank if allowed by forced_align, or a pad.
        # A single unknown token might be safer if blank is disallowed in targets.
        if char_to_id.get('<unk>', -1) != -1:
            targets_tensor = torch.tensor([[char_to_id['<unk>']]], dtype=torch.int32, device=device)
        else:  # Fallback if no unk, just use first valid label (e.g. pad)
            first_valid_label_id = next(iter(char_to_id.values()))
            targets_tensor = torch.tensor([[first_valid_label_id]], dtype=torch.int32, device=device)
    else:
        targets_tensor = torch.tensor([tokenized_transcript_chars], dtype=torch.int32, device=device)

    # --- Perform Forced Alignment ---
    # Ensure final_emissions is on the correct device for forced_align
    final_emissions = final_emissions.to(device)

    # The `blank` token ID depends on the model's vocabulary structure.
    # For torchaudio pipelines, it's often explicitly defined or can be found from bundle.
    # Typically, it's 0 for '<s>' or '<blank>' in some vocabs.
    # forced_align expects blank to be index, not token string.
    # MMS_FA typically uses <s> as blank=0. Or <blank> if present.
    # Let's try to find '<s>' or '<blank>' in labels.
    blank_id = -1
    if '<s>' in char_to_id:
        blank_id = char_to_id['<s>']
    elif '<blank>' in char_to_id:
        blank_id = char_to_id['<blank>']
    elif 0 < len(labels):
        blank_id = 0  # Fallback to first label if specific blank not found

    if blank_id == -1:
        logger.warning("Could not determine blank_id for forced_align. Defaulting to 0. This might be incorrect.")
        blank_id = 0

    # Ensure targets do not contain the blank_id (as per torchaudio.functional.forced_align requirements)
    # This should be handled by the tokenization above if blank_id is a special token not part of actual text.
    # If tokenized_transcript_chars could contain blank_id, filter it.
    # However, typical text won't map to blank.

    if final_emissions.shape[1] == 0:  # No emission frames
        logger.warning("Zero emission frames after stitching. Cannot perform alignment.")
        aligned_tokens_frames = torch.empty(0, dtype=torch.int64)
        alignment_scores_frames = torch.empty(0, dtype=torch.float32)
    elif targets_tensor.shape[1] == 0:
        logger.warning("Zero target tokens. Cannot perform alignment.")
        aligned_tokens_frames = torch.empty(0, dtype=torch.int64)
        alignment_scores_frames = torch.empty(0, dtype=torch.float32)

    else:
        try:
            aligned_paths, alignment_scores_frames = F.forced_align(
                final_emissions,
                targets_tensor,
                blank=blank_id
            )  # input_lengths and target_lengths can be inferred if not provided for B=1
            aligned_tokens_frames = aligned_paths[0]  # Get the path for the first (only) batch item
        except Exception as e:
            logger.error(f"Error during torchaudio.functional.forced_align: {e}")
            logger.error(
                f"Emissions shape: {final_emissions.shape}, Targets shape: {targets_tensor.shape}, Blank ID: {blank_id}")
            # Return empty results or re-raise
            aligned_tokens_frames = torch.empty(0, dtype=torch.int64)
            alignment_scores_frames = torch.empty(0, dtype=torch.float32)

    # --- Post-process results (convert frame alignments to word timestamps) ---
    # This part requires mapping aligned token IDs back to characters/words and then to time.
    # The existing `_postprocess_results` in the original `__init__.py` can be adapted.
    # It needs:
    # 1. The original text units (words) that correspond to `targets_tensor`.
    #    The `normalized_transcript_text.split()` can serve as this if tokenization was word-based.
    # 2. Spans of characters from `aligned_tokens_frames` corresponding to each word.
    # 3. The emission `stride_sec` to convert frame indices to time.

    # For now, we'll return the raw frame alignments and scores.
    # The AlignmentTorch class will handle the detailed postprocessing into word timestamps.

    # Reconstruct transcript words for postprocessing (the ones that went into `targets_tensor`)
    # This should match the granularity of `lyrics_lines` used by subtitle generation
    processed_lyrics_lines = [line for line in raw_text_lines if line]  # Use original lines for final text output

    # The `_postprocess_results` function in subtitles.py will need:
    # - text (list of words/segments corresponding to targets_tensor)
    # - word_spans (derived from aligned_tokens_frames and how targets_tensor was made)
    # - waveform_tensor (for total duration reference if needed, or just use emission frames)
    # - final_emissions.shape[1] (total number of frames)
    # - model_bundle.sample_rate
    # - alignment_scores_frames
    # - model_output_stride_sec

    # To prepare for that, we need to map `aligned_tokens_frames` back to characters.
    # And then group these characters into the words from `normalized_transcript_text.split()`.

    # This simplified return will be processed by the calling class
    # It should return enough info for _postprocess_results to work.
    # Key parts: aligned_tokens_frames (indices), scores, model_output_stride_sec, original words

    # `F.merge_tokens` might be useful here if we have char-level alignments
    # It merges consecutive identical tokens (after mapping from ID to char)
    # However, we need word-level segments.

    # Placeholder for detailed results, to be filled by a more elaborate post-processing step:
    # The `AlignmentTorch` class will call a new post-processing function.

    # `load_transcript_from_file` returns (list_of_words, list_of_lines)
    # We need the list_of_words that corresponds to `targets_tensor`.
    # Let's use `normalized_transcript_text.split()` for now.
    transcript_words_for_postprocessing = normalized_transcript_text.split()
    if not transcript_words_for_postprocessing and normalized_transcript_text:  # single word, no spaces
        transcript_words_for_postprocessing = [normalized_transcript_text]

    # This is a simplified return. The `AlignmentTorch` class will refine it.
    # It now directly returns what `get_word_stamps` was expected to produce,
    # but with chunking implemented. The caller needs to adapt.
    # The original `get_word_stamps` returned: word_timestamps, model, lyrics_lines
    # We need to construct `word_timestamps` here or have `AlignmentTorch` do it.

    # Let's attempt to use the existing postprocessing structure.
    # This requires `token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])`
    # and `word_spans = unflatten(token_spans, [len(word) for word in transcript])`

    # Map aligned_tokens_frames (IDs) back to string tokens
    id_to_char = {i: char for i, char in enumerate(labels)}
    aligned_chars_str = [id_to_char.get(idx.item(), "<ERR>") for idx in aligned_tokens_frames]

    # `F.merge_tokens` operates on the raw path and scores from `forced_align`
    # The path is `aligned_tokens_frames`, scores are `alignment_scores_frames`
    if aligned_tokens_frames.numel() > 0:
        token_spans = F.merge_tokens(aligned_tokens_frames,
                                     alignment_scores_frames.squeeze(0))  # Squeeze scores if (1,T)
    else:
        token_spans = []  # empty list if no alignment

    # To use `unflatten`, we need `[len(word) for word in transcript_words_for_postprocessing]`
    # This assumes `token_spans` correspond to characters of those words.
    # This is true if `targets_tensor` was built from characters of these words.
    # (Adjusting for word_boundary_token '|')

    target_word_char_lengths = []
    temp_words = normalized_transcript_text.split()
    for word in temp_words:
        target_word_char_lengths.append(len(list(word)))  # Length of chars in the word

    if not token_spans:  # no alignment produced
        word_spans_for_postprocessing = [[Segment("<ERR>", 0, 0)] for _ in temp_words]  # dummy spans
    elif not target_word_char_lengths and token_spans:  # single word case, no split by space
        word_spans_for_postprocessing = [token_spans]
    elif not target_word_char_lengths and not token_spans:
        word_spans_for_postprocessing = []
    else:
        try:
            word_spans_for_postprocessing = unflatten(token_spans, target_word_char_lengths)
        except Exception as e:
            logger.error(
                f"Error during unflatten step for word spans: {e}. Lengths: {target_word_char_lengths}, token_spans len: {len(token_spans)}")
            # Fallback: create dummy spans
            word_spans_for_postprocessing = [[Segment("<ERR>", 0, 0)] for _ in temp_words]

    # Return structure similar to what `_postprocess_results` expects,
    # or let AlignmentTorch call it.
    # `_postprocess_results(text, spans, waveform, num_frames, sample_rate, scores)`
    # `text` is `transcript_words_for_postprocessing`
    # `spans` is `word_spans_for_postprocessing`
    # `waveform` is `waveform_tensor`
    # `num_frames` is `final_emissions.shape[1]`
    # `sample_rate` is `model_bundle.sample_rate`
    # `scores` is `alignment_scores_frames` (this needs to align with how spans are structured)

    # For now, let `AlignmentTorch` handle the call to `_postprocess_results` by returning these components.
    return (
        transcript_words_for_postprocessing,  # list of words
        word_spans_for_postprocessing,  # list of lists of Segments (char spans per word)
        final_emissions.shape[1],  # total_emission_frames
        alignment_scores_frames.cpu(),  # frame-level scores from forced_align
        model_output_stride_sec,  # calculated stride
        model,  # the model instance
        processed_lyrics_lines  # original text lines (cleaned)
    )