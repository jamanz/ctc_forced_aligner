import numpy
import math
import logging
from typing import Tuple

from .constants import SAMPLING_FREQ, VOCAB_DICT
from .utils import time_to_frame  # Assuming this is already in utils.py and handles stride correctly

logger = logging.getLogger(__name__)


def generate_emissions_onnx(
        session,  # ONNX InferenceSession
        audio_waveform: numpy.ndarray,
        window_length_sec: int = 30,
        context_length_sec: int = 2,  # This is 'context' on EACH side of the window
        batch_size: int = 4
) -> Tuple[numpy.ndarray, float]:
    """
    Generates CTC emissions from an audio waveform using an ONNX model.
    Processes audio in overlapping windows.
    Returns:
        emissions (numpy.ndarray): Log probabilities from the model.
        stride_sec (float): The time duration in seconds corresponding to one frame in the emissions.
    """
    if audio_waveform.ndim != 1:
        raise ValueError("audio_waveform must be a 1D array")

    # Convert seconds to samples
    context_samples = context_length_sec * SAMPLING_FREQ
    window_samples = window_length_sec * SAMPLING_FREQ

    # Calculate required padding for the entire waveform to be processed in windows
    # Ensure total length is a multiple of window_samples if processing strictly in window_samples blocks
    num_full_windows_for_padding = math.ceil(audio_waveform.shape[0] / window_samples)
    padded_length_for_windows = num_full_windows_for_padding * window_samples
    extension_samples = max(0, padded_length_for_windows - audio_waveform.shape[0])

    # Pad the waveform: context at start, context + extension at end
    padded_waveform = numpy.pad(audio_waveform, (context_samples, context_samples + extension_samples), mode="constant")

    # Calculate number of windows based on the main audio part (excluding initial context)
    # Each step is window_samples
    num_windows = (padded_waveform.shape[0] - 2 * context_samples - extension_samples) // window_samples
    if (padded_waveform.shape[
            0] - 2 * context_samples - extension_samples) % window_samples > 0:  # if there's a remainder
        num_windows += 1

    input_windows_list = []
    for i in range(num_windows):
        start_idx = i * window_samples  # Start of the main window part
        # The input to the model includes context on both sides
        # Model input: [context | window | context]
        model_input_start = start_idx
        model_input_end = start_idx + window_samples + 2 * context_samples

        # Ensure we don't go past the end of padded_waveform
        actual_model_input_end = min(model_input_end, padded_waveform.shape[0])
        current_window_input = padded_waveform[model_input_start:actual_model_input_end]

        # If the current window input is shorter than expected (window + 2*context), pad it
        expected_input_len = window_samples + 2 * context_samples
        if current_window_input.shape[0] < expected_input_len:
            padding_needed = expected_input_len - current_window_input.shape[0]
            current_window_input = numpy.pad(current_window_input, (0, padding_needed), mode='constant')

        input_windows_list.append(current_window_input)

    if not input_windows_list:  # Handle very short audio that doesn't even form one window.
        # Process the entire padded_waveform as one chunk if it's shorter than a standard model input window
        expected_input_len = window_samples + 2 * context_samples
        if padded_waveform.shape[0] < expected_input_len:
            padding_needed = expected_input_len - padded_waveform.shape[0]
            input_windows_list.append(numpy.pad(padded_waveform, (0, padding_needed), mode='constant'))
        else:  # It's long enough for one window, or slightly longer but not enough for num_windows logic above
            input_windows_list.append(padded_waveform[:expected_input_len])

    input_windows_np = numpy.array(input_windows_list)

    logger.debug(f"ONNX Inference: Processing {input_windows_np.shape[0]} windows, batch size: {batch_size}")

    emissions_list = []
    for i in range(0, input_windows_np.shape[0], batch_size):
        input_batch = input_windows_np[i: i + batch_size]
        onnx_inputs = {"input_values": input_batch.astype(numpy.float32)}
        # Assuming the ONNX model output name is "logits"
        # You might need to inspect your ONNX model to confirm output names, e.g., using Netron
        outputs = session.run(["logits"], onnx_inputs)
        emissions_list.append(outputs[0])

    raw_emissions_batched = numpy.concatenate(emissions_list,
                                              axis=0)  # Shape: (NumWindows, SeqLenWithContext, VocabSize)

    # Remove context frames from each window's emissions
    # The number of emission frames for context_length_sec needs to be estimated if not fixed
    # Let's assume a fixed stride for now, e.g., 20ms per frame for Wav2Vec2-like models
    # This is a simplification; ideally, the model's actual stride is used.
    # The original code used time_to_frame, implying a stride was known or calculated.
    # Let's try to calculate stride from one window pass if possible, or use a typical value.

    # Calculate stride dynamically based on first window's output (if non-empty)
    # This is more robust than assuming a fixed stride.
    if raw_emissions_batched.size > 0:
        single_window_input_samples = input_windows_np.shape[1]  # e.g., window_samples + 2 * context_samples
        single_window_output_frames = raw_emissions_batched.shape[1]  # SeqLenWithContext

        if single_window_output_frames == 0:  # Avoid division by zero if a window yields no frames
            logger.warning("ONNX model produced zero output frames for a window. Cannot determine stride.")
            # Fallback or raise error - for now, use a default that might be incorrect
            samples_per_output_frame = 320  # Typical for Wav2Vec2 at 16kHz (20ms frames)
        else:
            samples_per_output_frame = single_window_input_samples / single_window_output_frames
    else:  # No emissions produced at all (e.g. very short audio, no windows processed)
        logger.warning("ONNX model produced no emissions. Cannot determine stride.")
        samples_per_output_frame = 320  # Fallback

    stride_sec = samples_per_output_frame / SAMPLING_FREQ

    if stride_sec <= 0:  # Should not happen if samples_per_output_frame is valid
        logger.error(f"Calculated stride_sec is non-positive ({stride_sec}). Defaulting.")
        samples_per_output_frame = 320
        stride_sec = samples_per_output_frame / SAMPLING_FREQ

    logger.debug(
        f"Calculated samples_per_output_frame for ONNX: {samples_per_output_frame:.2f}, stride_sec: {stride_sec:.4f}s")

    context_frames = math.ceil(context_samples / samples_per_output_frame)

    processed_emissions_list = []
    for i in range(raw_emissions_batched.shape[0]):
        current_emissions = raw_emissions_batched[i, :, :]
        # Remove emissions corresponding to the left and right context
        # Shape of current_emissions: (SeqLenWithContext, VocabSize)
        if current_emissions.shape[0] > 2 * context_frames:
            # The central part of the emissions, corresponding to the 'window_samples'
            center_emissions = current_emissions[context_frames: -context_frames, :]
        else:  # Window is too short, probably only context or less
            # Take whatever is there, or decide how to handle. For now, take all if too short to trim.
            # This might happen if window_length_sec is very small.
            logger.warning(
                f"Emission window too short to trim context frames. Window frames: {current_emissions.shape[0]}, Context frames (each side): {context_frames}")
            center_emissions = current_emissions
        processed_emissions_list.append(center_emissions)

    # Concatenate emissions from all windows
    final_emissions = numpy.concatenate(processed_emissions_list, axis=0)  # (TotalFrames, VocabSize)

    # Remove extra padding frames that were added to the end of the audio_waveform
    # total_audio_frames_expected = math.ceil(audio_waveform.shape[0] / samples_per_output_frame)
    # if final_emissions.shape[0] > total_audio_frames_expected:
    #     final_emissions = final_emissions[:total_audio_frames_expected, :]

    # The original `extension / SAMPLING_FREQ` removal logic was based on time.
    # Let's convert extension_samples to frames to remove.
    extension_frames_to_remove = math.ceil(extension_samples / samples_per_output_frame)
    if extension_frames_to_remove > 0 and final_emissions.shape[0] > extension_frames_to_remove:
        final_emissions = final_emissions[:-extension_frames_to_remove, :]
    elif extension_frames_to_remove > 0 and final_emissions.shape[0] <= extension_frames_to_remove:
        logger.warning(
            f"Trying to remove too many extension frames ({extension_frames_to_remove}) from emissions of length {final_emissions.shape[0]}. Keeping all.")

    # Apply log softmax (matching PyTorch post-processing generally)
    exp_emissions = numpy.exp(final_emissions)
    sum_exp_emissions = numpy.sum(exp_emissions, axis=-1, keepdims=True)
    log_softmax_emissions = numpy.log(exp_emissions / (sum_exp_emissions + 1e-9))  # Add epsilon for stability

    # Add extra dimension for <star> token if vocabulary assumes it but model doesn't output it
    # This depends on the specific ONNX model and the vocabulary structure.
    # The original code added a column of zeros. Let's assume it's needed for compatibility.
    # The VOCAB_DICT has 31 entries. If model outputs 31, this might not be needed,
    # or if '<star>' is meant to be the 32nd, then yes.
    # If the model already outputs for all vocab entries including a potential star or equivalent, skip.
    # Check if model output size matches vocab size used by get_alignments
    if log_softmax_emissions.shape[-1] < len(VOCAB_DICT) + 1:  # +1 if star is added
        # This part is tricky and depends on how the ONNX model was trained/exported
        # and how `get_alignments` expects the vocab.
        # Original code added one column for <star>. Let's replicate that behavior.
        # It assumes the model's output corresponds to VOCAB_DICT excluding star,
        # and star is the last one.
        logger.debug(
            f"Adding dummy <star> emission column to ONNX output. Original shape: {log_softmax_emissions.shape}")
        star_emissions = numpy.full((log_softmax_emissions.shape[0], 1), -10.0,
                                    dtype=numpy.float32)  # Low log-prob for star
        log_softmax_emissions = numpy.concatenate([log_softmax_emissions, star_emissions], axis=1)

    logger.debug(f"ONNX Final Emissions dtype: {log_softmax_emissions.dtype}, shape: {log_softmax_emissions.shape}")
    return log_softmax_emissions.astype(numpy.float32), stride_sec