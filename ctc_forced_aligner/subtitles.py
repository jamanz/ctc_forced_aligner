import os
import math
import numpy
import logging
from typing import List, Dict, Any

from .utils import Segment, frame_to_time  # Assuming Segment and frame_to_time are in utils

logger = logging.getLogger(__name__)


def merge_subtitle_segments(segments: List[Dict[str, Any]], threshold_sec: float = 0.0):
    """Merges subtitle segments if the gap between them is less than threshold_sec."""
    if not segments:
        return

    for i in range(len(segments) - 1):
        # Ensure 'end' of current and 'start' of next are comparable
        current_end = segments[i].get("end", 0.0)
        next_start = segments[i + 1].get("start", 0.0)

        if next_start - current_end < threshold_sec:
            segments[i + 1]["start"] = current_end


def _postprocess_word_level_alignments(
        text_units: List[str],  # List of words or text segments that were aligned
        word_spans: List[List[Segment]],  # List of (list of char Segments for each text_unit)
        # Each Segment has start/end in EMISSION FRAMES
        total_emission_frames: int,
        model_output_stride_sec: float,  # Time duration of one emission frame
        # waveform_shape_for_ratio: Tuple, # e.g., (1, total_audio_samples) Not directly used if stride_sec is accurate
        # sample_rate: int, # Not directly used if stride_sec is accurate
        alignment_scores_per_frame: torch.Tensor  # Raw scores from F.forced_align (Frame-level)
) -> List[Dict[str, Any]]:
    """
    Post-processes alignment results (from frame-level to word-level timestamps).
    `word_spans`: for each word, a list of its character segments (label, start_frame, end_frame).
    """
    results = []
    if len(text_units) != len(word_spans):
        logger.warning(
            f"Mismatch between number of text units ({len(text_units)}) and word_spans ({len(word_spans)}). Postprocessing might be inaccurate.")
        # Attempt to process the minimum length
        min_len = min(len(text_units), len(word_spans))
        text_units = text_units[:min_len]
        word_spans = word_spans[:min_len]

    for i, unit_text in enumerate(text_units):
        char_segments_for_unit = word_spans[i]

        if not char_segments_for_unit or not isinstance(char_segments_for_unit[0], Segment):
            logger.warning(f"Word '{unit_text}' (index {i}) has no valid char_segments. Skipping.")
            # Add a placeholder or skip
            results.append({
                "start": results[-1]["end"] if results else 0.0,
                "end": results[-1]["end"] if results else 0.0,
                "text": unit_text,
                "score": -1000.0  # Indicate poor alignment
            })
            continue

        # Determine start and end frames for the entire word/unit
        # The Segments in char_segments_for_unit should already account for blank padding removal if get_spans was used.
        # word_start_frame = char_segments_for_unit[0].start
        # word_end_frame = char_segments_for_unit[-1].end

        # More robust: find min start and max end from all segments of the word
        # (especially if get_spans includes padding segments around the word)
        actual_char_segments = [s for s in char_segments_for_unit if
                                s.label not in ["<blank>", "<pad>", "<unk>", "</s>"]]  # Filter out padding/special

        if not actual_char_segments:  # Only padding/special segments for this word
            # Try to infer from original segments if they exist
            word_start_frame = char_segments_for_unit[0].start
            word_end_frame = char_segments_for_unit[-1].end
            # Ensure end >= start
            if word_end_frame < word_start_frame: word_end_frame = word_start_frame
        else:
            word_start_frame = min(s.start for s in actual_char_segments)
            word_end_frame = max(s.end for s in actual_char_segments)

        # Convert frame indices to time
        audio_start_sec = frame_to_time(word_start_frame, model_output_stride_sec)
        audio_end_sec = frame_to_time(word_end_frame + 1,
                                      model_output_stride_sec)  # +1 because frame indices are inclusive

        # Calculate score for the word
        # Sum of scores for frames within this word's span
        score = 0.0
        if alignment_scores_per_frame.numel() > 0 and word_end_frame < alignment_scores_per_frame.shape[0]:
            # Sum scores of the frames that constitute the word
            # This requires knowing which frames in alignment_scores_per_frame correspond to this word.
            # alignment_scores_per_frame is for *all* emission frames.
            # We need the sum of scores for frames from word_start_frame to word_end_frame.
            try:
                word_scores = alignment_scores_per_frame[word_start_frame: word_end_frame + 1]
                score = word_scores.sum().item()
            except IndexError:
                logger.warning(
                    f"IndexError accessing scores for word '{unit_text}'. Frames: {word_start_frame}-{word_end_frame}. Total score frames: {alignment_scores_per_frame.shape[0]}")
                score = -1000.0  # Error score
        elif alignment_scores_per_frame.numel() == 0 and total_emission_frames > 0:
            score = -1.0  # No scores available but alignment happened
        else:
            score = -1000.0  # Error or no frames

        results.append({
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": unit_text,
            "score": score
        })

    merge_subtitle_segments(results, threshold_sec=0.01)  # Small threshold to close tiny gaps
    return results


def _format_time_srt(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm"""
    millisec = round(seconds * 1000)
    sec = millisec // 1000
    ms = millisec % 1000
    minute = sec // 60
    sec %= 60
    hour = minute // 60
    minute %= 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    """Converts seconds to WebVTT time format HH:MM:SS.mmm"""
    millisec = round(seconds * 1000)
    sec = millisec // 1000
    ms = millisec % 1000
    minute = sec // 60
    sec %= 60
    hour = minute // 60
    minute %= 60
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def _generate_subtitle_file_content(
        word_timestamps: List[Dict[str, Any]],  # List of {"start": float, "end": float, "text": str}
        original_lyrics_lines: List[str],  # Raw lines from input file/text
        time_formatter_func,
        header: str = "",
        line_separator: str = "\n\n"
) -> str:
    """
    Generates SRT or WebVTT file content from word timestamps and original lyric lines.
    """
    if not word_timestamps:
        return header  # Return only header if no timestamps

    # Attempt to reconstruct line-level timestamps
    line_timestamps = []
    current_word_idx = 0

    for original_line_text in original_lyrics_lines:
        if not original_line_text.strip():  # Skip empty lines
            continue

        line_start_time = -1.0
        line_end_time = -1.0

        # Find words from word_timestamps that form this original_line_text
        # This requires careful matching. For simplicity, we'll assume a greedy match based on available words.
        # A more robust approach would be to normalize original_line_text and word_timestamps[text]
        # and perform a sub-sequence match or use the pre-normalized words if available.

        # For now, let's do a simpler line matching strategy based on word accumulation
        # This part needs to be robust. The original code in __init__ had a complex way.
        # We need a way to map words from word_timestamps back to the original_lyrics_lines.

        # Simplified strategy: Accumulate words until the current line is "covered".
        # This won't be perfect if word_timestamps[text] are heavily normalized vs original_lyrics_lines.

        temp_line_buffer = []
        words_in_current_line_segment = 0

        # Try to match words to form the current original_line_text
        # This is a complex problem if normalization differs significantly.
        # The original _generate_srt/_generate_webvtt had a loop that accumulated words.
        # Let's try to replicate that basic idea.

        start_word_idx_for_line = current_word_idx

        # This matching is very basic and might fail if text normalization is aggressive
        temp_reconstructed_line = ""
        last_word_in_line_idx = -1

        for k in range(start_word_idx_for_line, len(word_timestamps)):
            word_data = word_timestamps[k]
            # Normalize both original line and word for comparison
            normalized_original_line_lower = original_line_text.lower()
            # Assume word_data["text"] is already somewhat normalized

            # If temp_reconstructed_line + word_data["text"] is a prefix of normalized_original_line_lower
            # or closely matches it.
            next_reconstructed_candidate = (temp_reconstructed_line + " " + word_data["text"]).strip()

            if normalized_original_line_lower.startswith(next_reconstructed_candidate.lower()):
                temp_reconstructed_line = next_reconstructed_candidate
                if line_start_time < 0: line_start_time = word_data["start"]
                line_end_time = word_data["end"]
                last_word_in_line_idx = k
            else:
                # Word does not fit, previous sequence was the line
                break  # Stop accumulating for this original_line_text

        if line_start_time >= 0 and last_word_in_line_idx >= start_word_idx_for_line:
            line_timestamps.append({
                "start": line_start_time,
                "end": line_end_time,
                "text": original_line_text  # Use the raw original line
            })
            current_word_idx = last_word_in_line_idx + 1
        elif original_line_text:  # If no match found, but line exists, create a placeholder
            # This can happen if word timestamps don't align well with original lines
            # Default to a short duration or use previous end time
            prev_end = line_timestamps[-1]["end"] if line_timestamps else (
                word_timestamps[0]["start"] if word_timestamps else 0)
            logger.warning(f"Could not align words to original line: '{original_line_text}'. Creating placeholder.")
            line_timestamps.append({
                "start": prev_end,
                "end": prev_end + 0.1,  # Short duration
                "text": original_line_text
            })

    # Build file content
    subtitle_content_parts = [header] if header else []
    for i, entry in enumerate(line_timestamps, start=1):
        start_formatted = time_formatter_func(entry["start"])
        end_formatted = time_formatter_func(entry["end"])
        text = entry["text"].strip()
        # Capitalize first letter of the line
        if text:
            text = text[0].upper() + text[1:]

        subtitle_content_parts.append(f"{i}")
        subtitle_content_parts.append(f"{start_formatted} --> {end_formatted}")
        subtitle_content_parts.append(text)

    # Join with appropriate separators
    if header:  # Header implies content starts after it
        return header + line_separator.join(
            line_separator.join(subtitle_content_parts[i:i + 3]) for i in range(1, len(subtitle_content_parts), 3)
        )
    else:  # No header, direct content
        return line_separator.join(
            line_separator.join(subtitle_content_parts[i:i + 3]) for i in range(0, len(subtitle_content_parts), 3)
        )


def generate_srt_content(word_timestamps: List[Dict[str, Any]], original_lyrics_lines: List[str]) -> str:
    return _generate_subtitle_file_content(word_timestamps, original_lyrics_lines, _format_time_srt,
                                           line_separator="\n\n") + "\n\n"


def generate_webvtt_content(word_timestamps: List[Dict[str, Any]], original_lyrics_lines: List[str]) -> str:
    return _generate_subtitle_file_content(word_timestamps, original_lyrics_lines, _format_time_vtt,
                                           header="WEBVTT\n\n", line_separator="\n\n") + "\n\n"