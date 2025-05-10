import os

# Audio settings
SAMPLING_FREQ = 16000
# Default model URL for ONNX
DEFAULT_ONNX_MODEL_URL = 'https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx'
DEFAULT_MODEL_PATH = os.path.join(os.path.expanduser("~"), "ctc_forced_aligner", "model.onnx")

# Vocabulary for the default ONNX model's tokenizer
# Ensure this matches the tokenizer used by the ONNX model.
# For PyTorch models from torchaudio.pipelines, the vocabulary/tokenizer is handled by the bundle.
VOCAB_DICT = {
    '<blank>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'a': 4, 'i': 5, 'e': 6, 'n': 7, 'o': 8, 'u': 9, 't': 10,
    's': 11, 'r': 12, 'm': 13, 'k': 14, 'l': 15, 'd': 16, 'g': 17, 'h': 18, 'y': 19, 'b': 20, 'p': 21,
    'w': 22, 'c': 23, 'v': 24, 'j': 25, 'z': 26, 'f': 27, "'": 28, 'q': 29, 'x': 30
}

# For PyTorch chunking
DEFAULT_CHUNK_LENGTH_SEC = 20  # Process 20 seconds of audio at a time
DEFAULT_OVERLAP_LENGTH_SEC = 2  # Overlap by 2 seconds on each side for context
DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS = 4 # Batch size for chunks fed to PyTorch model