import re
import unicodedata
from typing import List, Tuple, Dict, Set

from .norm_config import norm_config as global_norm_config  # alias to avoid clash
from .constants import VOCAB_DICT


# Helper to build patterns from sets for regex
def _build_regex_pattern(char_set_str: str) -> str:
    return r"[" + char_set_str + r"]"


def text_normalize(
        text: str,
        iso_code: str = "eng",  # Default to English
        lower_case: bool = True,
        remove_numbers: bool = True,
        remove_brackets: bool = False  # Keep brackets by default
) -> str:
    """Normalizes text based on language-specific rules and general configurations."""
    config = global_norm_config.get(iso_code, global_norm_config["*"])  # Get lang-specific or default

    # Ensure all necessary fields are present, falling back to default config if not
    for field in ["lower_case", "punc_set", "del_set", "mapping", "digit_set", "unicode_norm", "rm_diacritics"]:
        if field not in config:
            config[field] = global_norm_config["*"][field]

    # 1. Unicode Normalization
    text = unicodedata.normalize(config["unicode_norm"], text)

    # 2. Lowercasing
    if config["lower_case"] and lower_case:
        text = text.lower()

    # 3. Handle Brackets (especially with numbers, e.g., verse numbers)
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)  # Remove brackets containing numbers
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)  # Remove all other brackets if flag is set

    # 4. Apply Character Mappings
    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # 5. Replace Punctuation with Space
    punct_pattern = _build_regex_pattern(config["punc_set"])
    normalized_text = re.sub(punct_pattern, " ", text)

    # 6. Remove Characters in Deletion List
    delete_pattern = _build_regex_pattern(config["del_set"])
    normalized_text = re.sub(delete_pattern, "", normalized_text)

    # 7. Remove Numbers (if flag is set)
    if remove_numbers:
        digits_pattern = _build_regex_pattern(config["digit_set"]) + "+"
        # Regex to match standalone numbers (surrounded by spaces or start/end of string)
        standalone_digits_pattern = (
                r"(^" + digits_pattern + r"(?=\s|$)|(?<=\s)" + digits_pattern + r"(?=\s|$)|(?<=\s)" + digits_pattern + r"$)"
        )
        normalized_text = re.sub(standalone_digits_pattern, " ", normalized_text)

    # 8. Remove Diacritics (if configured)
    if config["rm_diacritics"]:
        # Ensure unidecode is available if this option is used
        try:
            from unidecode import unidecode
            normalized_text = unidecode(normalized_text)
        except ImportError:
            pass  # Silently skip if unidecode is not installed

    # 9. Remove Extra Spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text


def normalize_uroman(text: str) -> str:
    """Basic uroman-like normalization for romanized text."""
    text = text.lower()
    text = re.sub(r"([^a-z' ])", " ", text)  # Keep only lowercase letters, apostrophes, and spaces
    text = re.sub(r" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts: List[str], iso_code: str = None) -> List[str]:
    """
    Converts normalized transcripts to a 'uromanized' representation (basic unidecode and char spacing).
    The `iso_code` is not strictly used here for uroman rules but kept for API consistency.
    """
    try:
        from unidecode import unidecode
    except ImportError:
        raise ImportError("The 'unidecode' library is required for romanization. Please install it.")

    uroman_texts = []
    for text_line in norm_transcripts:
        romanized_line = unidecode(text_line)
        # Split characters and add spaces, then normalize spaces
        char_spaced_line = " ".join(list(romanized_line.strip()))
        char_spaced_line = re.sub(r"\s+", " ", char_spaced_line).strip()
        uroman_texts.append(normalize_uroman(char_spaced_line))  # Further clean uroman output
    return uroman_texts


def split_text_into_units(text: str, unit: str = "word") -> List[str]:
    """Splits text into sentences, words, or characters."""
    if unit == "sentence":
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is required for sentence splitting. Please install it (`pip install nltk`) and download 'punkt'.")
        # Ensure 'punkt' tokenizer models are downloaded for NLTK
        try:
            return sent_tokenize(text)
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
            return sent_tokenize(text)
    elif unit == "word":
        return text.split()
    elif unit == "char":
        return list(text)
    else:
        raise ValueError("Split unit must be 'sentence', 'word', or 'char'")


def preprocess_transcript_for_alignment(
        text_content: str,
        language: str = "eng",
        romanize: bool = True,  # Typically True for the existing ONNX model
        split_unit: str = "word",
        star_insertion_strategy: str = "segment"  # "segment", "edges", or "none"
) -> Tuple[List[str], List[str]]:
    """
    Preprocesses text for alignment: normalization, (optional) romanization, token splitting, and <star> token insertion.
    Returns:
        tokens_for_model: List of strings, space-separated characters, ready for vocab mapping.
        text_units_starred: List of original text units (words/sentences) with <star> tokens.
    """
    assert split_unit in ["sentence", "word", "char"], "Split unit must be sentence, word, or char"
    assert star_insertion_strategy in ["segment", "edges", "none"], "Star strategy must be segment, edges, or none"

    # For certain languages, character-based splitting is more appropriate
    if language in ["jpn", "chi", "kor"]:  # Add other relevant languages
        split_unit = "char"

    original_text_units = split_text_into_units(text_content, split_unit)

    normalized_units = [text_normalize(unit.strip(), language) for unit in original_text_units]
    # Filter out empty strings that might result from normalization
    normalized_units = [unit for unit in normalized_units if unit]
    original_text_units = [unit for unit in original_text_units if
                           text_normalize(unit.strip(), language)]  # keep original units aligned

    if romanize:
        # `get_uroman_tokens` expects a list of strings and returns a list of space-separated chars
        tokens_for_model_intermediate = get_uroman_tokens(normalized_units, language)
    else:
        # For non-romanized, convert each normalized unit (word/char) into space-separated characters
        tokens_for_model_intermediate = [" ".join(list(unit)) for unit in normalized_units]
        tokens_for_model_intermediate = [re.sub(r"\s+", " ", token).strip() for token in tokens_for_model_intermediate]

    # Add <star> token based on strategy
    tokens_final_for_model = []
    text_units_final_starred = []

    if star_insertion_strategy == "segment":
        for i, token_model_repr in enumerate(tokens_for_model_intermediate):
            tokens_final_for_model.extend(["<star>", token_model_repr])
            text_units_final_starred.extend(["<star>", original_text_units[i]])
        if not tokens_final_for_model:  # Handle empty input by adding a single star if segment strategy
            tokens_final_for_model.append("<star>")
            text_units_final_starred.append("<star>")

    elif star_insertion_strategy == "edges":
        tokens_final_for_model = ["<star>"] + tokens_for_model_intermediate + ["<star>"]
        text_units_final_starred = ["<star>"] + original_text_units + ["<star>"]

    elif star_insertion_strategy == "none":
        tokens_final_for_model = tokens_for_model_intermediate
        text_units_final_starred = original_text_units

    # Ensure no empty strings in the final list for the model, replace with <unk> or handle as needed
    tokens_final_for_model = [tok if tok else "<unk>" for tok in tokens_final_for_model]

    return tokens_final_for_model, text_units_final_starred


def load_transcript_from_file(
        file_path: str,
        target_dictionary: Dict[str, int] = VOCAB_DICT
) -> Tuple[List[str], List[str]]:
    """
    Loads transcript from a file, cleans words based on the target_dictionary.
    Used primarily by the PyTorch `get_word_stamps` (original version).
    Returns:
        transcript: List of cleaned words.
        lyrics_lines: List of cleaned lines.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lyrics_lines_cleaned = []
    transcript_words_cleaned = []

    for line in lines:
        words = line.strip().lower().split()
        # Clean words: keep only characters present in the dictionary and not blank/pad
        cleaned_words_for_line = []
        for word in words:
            cleaned_word = "".join([char for char in word if
                                    char in target_dictionary and target_dictionary[char] > VOCAB_DICT[
                                        '<unk>']])  # Assuming 0,1,2,3 are special
            if cleaned_word:
                cleaned_words_for_line.append(cleaned_word)

        if cleaned_words_for_line:
            transcript_words_cleaned.extend(cleaned_words_for_line)
            lyrics_lines_cleaned.append(" ".join(cleaned_words_for_line))

    return transcript_words_cleaned, lyrics_lines_cleaned