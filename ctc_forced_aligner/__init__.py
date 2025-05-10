import logging
import os

# Configure basic logging for the package
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default log level, can be changed by user

logger.debug(f"ctc_forced_aligner package initializing from: {__file__}")

# Expose the main alignment classes as the public API
from .alignment import Alignment, AlignmentSingleton, AlignmentTorch, AlignmentTorchSingleton

# Expose constants that might be useful for users
from .constants import (
    SAMPLING_FREQ,
    DEFAULT_ONNX_MODEL_URL,
    DEFAULT_MODEL_PATH,
    VOCAB_DICT,
    DEFAULT_CHUNK_LENGTH_SEC,
    DEFAULT_OVERLAP_LENGTH_SEC,
    DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS
)

# Expose underlying utilities only if intended for advanced users, otherwise keep them internal.
# For now, let's keep them internal to the submodules.

__all__ = [
    'Alignment',
    'AlignmentSingleton',
    'AlignmentTorch',
    'AlignmentTorchSingleton',
    'SAMPLING_FREQ', # Exposing common constants
    'DEFAULT_ONNX_MODEL_URL',
    'DEFAULT_MODEL_PATH',
    'VOCAB_DICT',
    'DEFAULT_CHUNK_LENGTH_SEC',
    'DEFAULT_OVERLAP_LENGTH_SEC',
    'DEFAULT_PYTORCH_BATCH_SIZE_CHUNKS'
]

# Version of the package - can be managed here or via setup.py / other tools
__version__ = "1.0.3" # Assuming this refactor is a new version

logger.info(f"ctc_forced_aligner version {__version__} loaded.")