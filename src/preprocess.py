import nltk
nltk.download('stopwords')
import re
from typing import Iterable, List, Optional

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# --------- Compile reusable regex patterns ----------
# Arabic diacritics (tashkeel) ranges
_RE_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
# tatweel (kashida)
_RE_TATWEEL = re.compile(r"\u0640")
# keep Arabic letters + Arabic/Western digits + whitespace; drop everything else
_RE_NON_ARABIC_CHARS = re.compile(r"[^\u0600-\u06FF0-9\s]")

# optional: collapse extra spaces
_RE_SPACES = re.compile(r"\s+")

# Arabic punctuation to remove (you can extend)
AR_PUNCT = "،؛؟«»…ـ٪٫٬“”‘’"
_LATIN_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_RE_PUNCT = re.compile("[" + re.escape(AR_PUNCT + _LATIN_PUNCT) + "]")

# --------- Stopwords ----------
_NLTK_AR_STOPS = set(stopwords.words("arabic"))

# You can extend/trim this as needed for your domain
_EXTRA_AR_STOPS = {
    "هذ", "هذا", "هذه", "ذلك", "تلك", "لكن", "أو", "و", "في", "على", "الى", "إلى",
    "كما", "أيضا", "من", "عن", "ما", "مما", "مع", "قد", "كل", "ضمن", "كان", "كانت",
}
AR_STOPWORDS = (_NLTK_AR_STOPS | _EXTRA_AR_STOPS)


def normalize_arabic(
    text: str,
    *,
    normalize_hamza=True,
    normalize_ya=True,
    normalize_alif_maqsura=True,
    normalize_ta_marbuta="keep",  # "keep" | "h" (ة→ه) | "t" (ة→ت)
    remove_tatweel=True,
    remove_punct=True,
    keep_digits=True,
    keep_spaces=True,
    drop_non_arabic=True,
) -> str:
    """
    Light, practical normalization for Arabic text classification.
    """
    if text is None:
        return ""
    s = str(text)

    # Remove diacritics early (often desired before other maps)
    s = _RE_DIACRITICS.sub("", s)

    # Remove tatweel
    if remove_tatweel:
        s = _RE_TATWEEL.sub("", s)

    # Unify hamza/alif forms: أ إ آ ٱ -> ا
    if normalize_hamza:
        s = re.sub(r"[إأآٱ]", "ا", s)

    # Normalize alif maqsura ى -> ي
    if normalize_alif_maqsura:
        s = s.replace("ى", "ي")

    # Normalize dotless yaa (Persian/Urdu) to Arabic ي
    if normalize_ya:
        s = s.replace("ی", "ي")  # Farsi ya → Arabic ya

    # Taa marbuta handling
    if normalize_ta_marbuta == "h":
        s = s.replace("ة", "ه")
    elif normalize_ta_marbuta == "t":
        s = s.replace("ة", "ت")
    # else keep as is

    # Remove punctuation
    if remove_punct:
        s = _RE_PUNCT.sub(" ", s)

    # Drop non-Arabic chars (keeps Arabic letters + digits + spaces)
    if drop_non_arabic:
        s = _RE_NON_ARABIC_CHARS.sub(" ", s)

    # Optionally remove digits
    if not keep_digits:
        s = re.sub(r"\d+", " ", s)

    # Keep or collapse spaces
    s = _RE_SPACES.sub(" ", s) if keep_spaces else s.replace(" ", "")
    return s.strip()


def simple_tokenize(text: str) -> List[str]:
    # whitespace tokenization after normalization is often enough for bag-of-words models
    return [t for t in text.split() if t]


def remove_stopwords(tokens: Iterable[str], stopset: Optional[set] = None) -> List[str]:
    if stopset is None:
        stopset = AR_STOPWORDS
    return [t for t in tokens if t not in stopset]


def stem_isri(tokens: Iterable[str]) -> List[str]:
    stemmer = ISRIStemmer()
    # ISRI expects undiacritized input; we already removed diacritics in normalize_arabic
    return [stemmer.stem(t) for t in tokens]


# Optional: lemmatization via camel_tools (heavier, slower)
try:
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.morphology.lemmatizer import Lemmatizer
    _CAMEL_OK = True
    _mle = MLEDisambiguator.pretrained()
    _lem = Lemmatizer.pretrained()
except Exception:
    _CAMEL_OK = False
    _mle = None
    _lem = None

def lemmatize_camel(tokens: Iterable[str]) -> List[str]:
    if not _CAMEL_OK:
        # fall back to ISRI if camel_tools not available
        return stem_isri(tokens)
    sent = " ".join(tokens)
    disamb = _mle.disambiguate(simple_word_tokenize(sent))
    # pick first lemma per token (you can refine)
    return [ana.analyses[0].lemma if ana.analyses else ana.word for ana in disamb]


def preprocess_text(
    text: str,
    *,
    use_lemmatizer=False,   # if True and camel_tools available -> lemmatize; else ISRI stem
    remove_stops=True,
    custom_stopset: Optional[set] = None,
    **normalize_kwargs,
) -> List[str]:
    """
    Full pipeline: normalize → tokenize → stopwords → stem/lemma.
    Returns a list of processed tokens.
    """
    norm = normalize_arabic(text, **normalize_kwargs)
    toks = simple_tokenize(norm)
    if remove_stops:
        toks = remove_stopwords(toks, stopset=custom_stopset)
    toks = lemmatize_camel(toks) if use_lemmatizer else stem_isri(toks)
    return toks


def preprocess_series_to_text(
    s: pd.Series,
    *,
    use_lemmatizer=False,
    remove_stops=True,
    join_with_space=True,
    **normalize_kwargs,
) -> pd.Series:
    """
    Apply preprocess_text to a pandas Series. Optionally join tokens back to a string.
    """
    out = s.astype(str).map(
        lambda x: preprocess_text(
            x,
            use_lemmatizer=use_lemmatizer,
            remove_stops=remove_stops,
            **normalize_kwargs,
        )
    )
    return out.map(lambda toks: " ".join(toks)) if join_with_space else out
