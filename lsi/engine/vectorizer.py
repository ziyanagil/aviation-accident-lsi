from collections import Counter

import numpy as np

from lsi.schemas import TfidfState

TF_MODES = ("raw", "log")


class TfidfVectorizer:
    def __init__(self, tf_mode: str = "log", min_df: int = 1):
        if tf_mode not in TF_MODES:
            raise ValueError(f"tf_mode harus salah satu dari {TF_MODES}")
        if min_df < 1:
            raise ValueError("min_df minimal 1")
        self.tf_mode = tf_mode
        self.min_df = min_df
        self.vocab: list[str] = []
        self.term_index: dict[str, int] = {}
        self.idf: np.ndarray = np.empty(0)

    @classmethod
    def from_state(cls, state: TfidfState, tf_mode: str, min_df: int) -> "TfidfVectorizer":
        vectorizer = cls(tf_mode, min_df)
        vectorizer._set_vocab(state.vocab)
        vectorizer.idf = state.idf
        return vectorizer

    def _set_vocab(self, vocab: list[str]) -> None:
        self.vocab = list(vocab)
        self.term_index = {term: i for i, term in enumerate(self.vocab)}

    def _apply_tf(self, counts: np.ndarray) -> np.ndarray:
        counts = counts.astype(float)
        if self.tf_mode == "raw":
            return counts
        weighted = np.zeros_like(counts)
        positive = counts > 0
        weighted[positive] = 1.0 + np.log(counts[positive])
        return weighted

    def _count_matrix(self, docs_tokens: list[list[str]]) -> np.ndarray:
        counts = np.zeros((len(self.vocab), len(docs_tokens)))
        for j, tokens in enumerate(docs_tokens):
            for term, count in Counter(tokens).items():
                index = self.term_index.get(term)
                if index is not None:
                    counts[index, j] = count
        return counts

    def fit(self, docs_tokens: list[list[str]]) -> TfidfState:
        n_docs = len(docs_tokens)
        doc_freq = Counter(term for tokens in docs_tokens for term in set(tokens))
        vocab = sorted(t for t, df in doc_freq.items() if df >= self.min_df)
        if not vocab:
            raise ValueError("Vocabulary kosong setelah penerapan min_df")
        self._set_vocab(vocab)
        df_vector = np.array([doc_freq[t] for t in self.vocab], dtype=float)
        self.idf = np.log(n_docs / df_vector) + 1.0
        matrix = self._apply_tf(self._count_matrix(docs_tokens)) * self.idf[:, None]
        return TfidfState(vocab=self.vocab, idf=self.idf, matrix=matrix)

    def transform_query(self, tokens: list[str]) -> np.ndarray:
        counts = self._count_matrix([tokens])[:, 0]
        return self._apply_tf(counts) * self.idf
