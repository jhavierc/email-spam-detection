"""Hand-crafted text features used for exploratory data analysis."""
import re
import string

import pandas as pd

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b\w+\.(?:com|co\.uk|net|org)\b)", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")
CURRENCY_RE = re.compile(r"[£$€]|\b(?:gbp|usd|eur|pounds?)\b", re.I)
TOKEN_RE = re.compile(r"[a-z']+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were", "be", "been", "am",
    "i", "me", "my", "you", "your", "u", "ur", "he", "she", "it", "we", "they", "them", "him", "her",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "as", "so", "do", "did", "does", "not",
    "no", "this", "that", "these", "those", "have", "has", "had", "will", "can", "just", "what", "how",
    "when", "there", "then", "its", "it's", "i'm", "im", "up", "out", "get", "got", "go", "all", "now",
    "call", "know", "like", "come", "ok", "4", "2", "s", "t", "m", "ll", "d", "re", "ve", "lt", "gt",
    "dont", "don't", "our", "us", "about", "one", "only", "any", "who", "too", "also", "would", "should",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def extract_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of ``df`` enriched with numeric features describing each message."""
    out = df.copy()
    s = out[text_col]
    out["n_chars"] = s.str.len()
    out["n_words"] = s.str.split().str.len()
    out["n_sentences"] = s.str.count(r"[.!?]+") + 1
    out["avg_word_len"] = out["n_chars"] / out["n_words"].clip(lower=1)
    out["n_uppercase"] = s.str.count(r"[A-Z]")
    out["uppercase_ratio"] = out["n_uppercase"] / out["n_chars"].clip(lower=1)
    out["n_digits"] = s.str.count(r"\d")
    out["digit_ratio"] = out["n_digits"] / out["n_chars"].clip(lower=1)
    out["n_punctuation"] = s.apply(lambda t: sum(c in string.punctuation for c in t))
    out["n_exclamations"] = s.str.count("!")
    out["has_url"] = s.str.contains(URL_RE).astype(int)
    out["has_phone"] = s.str.contains(PHONE_RE).astype(int)
    out["has_currency"] = s.str.contains(CURRENCY_RE).astype(int)
    return out


NUMERIC_FEATURES = [
    "n_chars", "n_words", "n_sentences", "avg_word_len", "n_uppercase", "uppercase_ratio",
    "n_digits", "digit_ratio", "n_punctuation", "n_exclamations", "has_url", "has_phone", "has_currency",
]
