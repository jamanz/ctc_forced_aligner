import librosa
import numpy
import math
import os
import requests
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from .constants import SAMPLING_FREQ

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    label: str
    start: int  # frame index
    end: int  # frame index

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def load_audio(audio_file: str, target_sr: int = SAMPLING_FREQ, mono: bool = True) -> numpy.ndarray:
    """Loads audio file as a numpy array."""
    waveform, sr = librosa.load(audio_file, sr=None, mono=mono)  # Load with original SR
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return waveform.astype(numpy.float32)


def time_to_frame(time_sec: float, stride_sec: float) -> int:
    """Converts time in seconds to frame index."""
    if stride_sec <= 0:
        raise ValueError("Stride must be positive.")
    return int(time_sec / stride_sec)


def frame_to_time(frame_idx: int, stride_sec: float) -> float:
    """Converts frame index to time in seconds."""
    return frame_idx * stride_sec


def merge_repeats(path: List[int], idx_to_token_map: Dict[int, str]) -> List[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def get_spans(tokens: List[str], segments: List[Segment], blank_token: str) -> List[List[Segment]]:
    """
    Aligns token sequences with CTC segments to get spans for each token/word.
    Assumes tokens are words, and segments are character-level CTC outputs.
    """
    ltr_idx = 0
    tokens_idx = 0
    intervals = []  # Stores (start_segment_idx, end_segment_idx) for each token
    current_token_char_list = []

    for seg_idx, seg in enumerate(segments):
        if tokens_idx >= len(tokens):
            # All tokens processed, remaining segments should be blank or EOS
            if seg.label != blank_token and seg.label != '</s>':  # Assuming </s> is an EOS token
                logger.debug(f"Extra segment after all tokens processed: {seg.label}")
            continue

        if not current_token_char_list:
            if not tokens[tokens_idx].strip():  # Handle empty strings in tokens list (e.g. from multiple spaces)
                intervals.append((seg_idx, seg_idx - 1))  # Empty token, zero duration span
                tokens_idx += 1
                if tokens_idx >= len(tokens): continue
            current_token_char_list = list(
                tokens[tokens_idx].replace(" ", ""))  # Remove spaces if words are like "w o r d"

        # Skip blank segments from CTC
        if seg.label == blank_token:
            continue

        # Handle cases where segment label might be a special token not in the character list
        if seg.label not in VOCAB_DICT or VOCAB_DICT.get(seg.label, -1) < VOCAB_DICT['a']:  # Non-alphabetic or special
            if ltr_idx == 0 and len(current_token_char_list) == 0:  # Start of a new token which is empty
                pass  # Will be handled by the empty token logic above or next non-blank
            elif ltr_idx > 0:  # If we were in middle of a token, this might be an issue or inter-word symbol
                logger.debug(f"Special segment {seg.label} encountered mid-token {tokens[tokens_idx]}.")
            continue

        if ltr_idx < len(current_token_char_list) and seg.label == current_token_char_list[ltr_idx]:
            if ltr_idx == 0:  # Start of a new token match
                start_seg_interval_idx = seg_idx

            ltr_idx += 1

            if ltr_idx == len(current_token_char_list):  # Current token fully matched
                intervals.append((start_seg_interval_idx, seg_idx))
                tokens_idx += 1
                ltr_idx = 0
                current_token_char_list = []
        else:
            # Mismatch: This indicates an issue with alignment or transcript
            # It could be an extra character from CTC, or a transcript mismatch
            # For robustnes, we might decide to "skip" this segment or current token char
            logger.warning(
                f"Mismatch: Token '{tokens[tokens_idx]}', char_idx {ltr_idx} ('{current_token_char_list[ltr_idx] if ltr_idx < len(current_token_char_list) else 'OOB'}') "
                f"!= Segment '{seg.label}' at seg_idx {seg_idx}. Attempting to resync."
            )
            # Simple resync: if the current segment matches the START of the current token, reset.
            if len(current_token_char_list) > 0 and seg.label == current_token_char_list[0]:
                ltr_idx = 1  # Matched the first char
                start_seg_interval_idx = seg_idx
                if ltr_idx == len(current_token_char_list):  # Single char token
                    intervals.append((start_seg_interval_idx, seg_idx))
                    tokens_idx += 1
                    ltr_idx = 0
                    current_token_char_list = []
            # else, we effectively skip this CTC segment and hope the next one aligns.
            # Or, if stuck on a token char, advance to next token? More complex.

    # Check if all tokens were processed
    if tokens_idx < len(tokens):
        logger.warning(f"Not all tokens were aligned. Last aligned token index: {tokens_idx - 1} out of {len(tokens)}")
        # Add placeholder for remaining tokens if needed
        for i in range(tokens_idx, len(tokens)):
            last_seg_end = intervals[-1][1] if intervals else len(segments) - 1
            intervals.append((last_seg_end + 1, last_seg_end))  # empty spans for unaligned tokens

    # Construct spans based on intervals
    final_spans = []
    for start_idx, end_idx in intervals:
        if start_idx > end_idx:  # Handle empty tokens by creating a zero-length span
            if start_idx < len(segments):
                final_spans.append([Segment(blank_token, segments[start_idx].start, segments[start_idx].start - 1)])
            elif final_spans:  # If it's beyond segments, use previous span's end.
                final_spans.append([Segment(blank_token, final_spans[-1][-1].end + 1, final_spans[-1][-1].end)])
            else:  # No segments and no previous spans, default to 0, -1
                final_spans.append([Segment(blank_token, 0, -1)])

        else:
            span_segments = segments[start_idx: end_idx + 1]
            # Add padding from surrounding blank segments
            # Left padding
            if start_idx > 0:
                prev_seg = segments[start_idx - 1]
                if prev_seg.label == blank_token:
                    pad_start_frame = prev_seg.start if not final_spans else math.ceil(
                        (prev_seg.start + prev_seg.end) / 2)
                    # Ensure pad_start_frame is not after the current span's start
                    current_span_actual_start_frame = span_segments[0].start
                    pad_start_frame = min(pad_start_frame, current_span_actual_start_frame)
                    if pad_start_frame < current_span_actual_start_frame:
                        span_segments = [Segment(blank_token, pad_start_frame,
                                                 current_span_actual_start_frame - 1)] + span_segments
            # Right padding
            if end_idx + 1 < len(segments):
                next_seg = segments[end_idx + 1]
                if next_seg.label == blank_token:
                    pad_end_frame = next_seg.end if len(final_spans) == len(intervals) - 1 else math.floor(
                        (next_seg.start + next_seg.end) / 2)
                    current_span_actual_end_frame = span_segments[-1].end
                    # Ensure pad_end_frame is not before the current span's end
                    pad_end_frame = max(pad_end_frame, current_span_actual_end_frame)
                    if pad_end_frame > current_span_actual_end_frame:
                        span_segments = span_segments + [
                            Segment(blank_token, current_span_actual_end_frame + 1, pad_end_frame)]
            final_spans.append(span_segments)

    return final_spans


def unflatten(list_, lengths: List[int]) -> List[List[Any]]:
    i, ret = 0, []
    for length in lengths:
        ret.append(list_[i: i + length])
        i += length
    return ret


def ensure_model_downloaded(model_path: str, url: str):
    """Downloads the model file if it does not exist locally."""
    if not os.path.exists(model_path):
        logger.info(f"Downloading model from {url} to {model_path}...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()  # Raise an exception for bad status codes
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download the model: {e}")
            raise Exception(f"Failed to download the model from {url}. Error: {e}")