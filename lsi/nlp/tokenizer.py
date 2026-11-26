import re
from functools import lru_cache

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None

TOKEN_RE = re.compile(r"[a-z]+")
FALLBACK_SUFFIXES = ("ingly", "edly", "ing", "ed", "es", "ly", "s")
MIN_TOKEN_LENGTH = 3

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers him his how i if
in into is it its itself just me more most my no nor not of off on once only or
other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your
""".split())


@lru_cache(maxsize=None)
def stem_word(word: str) -> str:
    if _STEMMER is not None:
        return _STEMMER.stem(word)
    for suffix in FALLBACK_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_TOKEN_LENGTH:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= MIN_TOKEN_LENGTH]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_word(t) for t in tokens]


def preprocess(text: str, use_stemming: bool = True) -> list[str]:
    tokens = remove_stopwords(tokenize(text))
    return stem_tokens(tokens) if use_stemming else tokens
