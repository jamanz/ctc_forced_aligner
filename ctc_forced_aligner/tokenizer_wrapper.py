import numpy
from .constants import VOCAB_DICT

class Tokenizer:
    """Standalone Tokenizer primarily for the ONNX Wav2Vec2 CTC Forced Alignment"""

    def __init__(self, vocab_dict=None):
        self.vocab_dict = vocab_dict if vocab_dict else VOCAB_DICT
        self.id_to_token = {v: k for k, v in self.vocab_dict.items()}
        # Special tokens
        self.PAD_TOKEN_ID = self.vocab_dict.get("<pad>", 1) # Default pad for ONNX
        self.BLANK_TOKEN_ID = self.vocab_dict.get("<blank>", 0) # Default blank for ONNX
        self.UNK_TOKEN_ID = self.vocab_dict.get("<unk>", 3)

    def encode(self, text):
        """Convert text to token IDs."""
        text = text.lower().strip()
        token_ids = [self.vocab_dict.get(char, self.UNK_TOKEN_ID) for char in text]
        return numpy.array(token_ids, dtype=numpy.int64)

    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return "".join([self.id_to_token.get(i, "?") for i in token_ids])

    def get_vocab(self):
        """Return the vocabulary dictionary."""
        return self.vocab_dict

    @property
    def pad_token_id(self):
        return self.PAD_TOKEN_ID

    @property
    def blank_token_id(self):
        return self.BLANK_TOKEN_ID

    @property
    def unk_token_id(self):
        return self.UNK_TOKEN_ID

    def get_dict_for_torchaudio(self):
        """
        Returns a dictionary compatible with torchaudio pipelines
        (e.g. for MMS_FA, where labels are characters).
        This might need adjustment if models expect '|' separated tokens or other formats.
        """
        # For character-based models like MMS_FA, the direct vocab might be suitable.
        # However, torchaudio bundles.MMS_FA.get_dict() returns something like:
        # {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'e': 5, ...}
        # The ctc_forced_aligner's VOCAB_DICT is char-based without '|', so it's more for the custom ONNX model.
        # For torchaudio pipelines, it's best to use `bundle.get_dict()`
        raise NotImplementedError("Use bundle.get_dict() for torchaudio pipeline models.")