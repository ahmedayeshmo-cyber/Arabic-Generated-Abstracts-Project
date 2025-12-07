# utils.py
import os
import joblib

#  TOKENIZATION UTILITIES

def simple_word_tokenize(text: str) -> list[str]:
    """Tokenize into Arabic words, Latin words, and punctuation."""
    # Uses re2 if available, otherwise standard re
    return _re.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=_re.VERSION1)


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences using standard and Arabic punctuation."""
    text = str(text)
    # Split after sentence-ending punctuation followed by whitespace
    parts = _re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def paragraph_tokenize(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines."""
    if not isinstance(text, str):
        return []
    paragraphs = _re.split(r'\s*\n\s*\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

