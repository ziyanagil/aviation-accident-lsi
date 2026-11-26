from pathlib import Path

import numpy as np
import pandas as pd

from lsi import config
from lsi.engine.similarity import cosine_to_rows
from lsi.engine.svd_model import LSIModel
from lsi.engine.vectorizer import TfidfVectorizer
from lsi.io.loader import compute_fingerprint, load_cache, save_cache
from lsi.nlp.tokenizer import preprocess
from lsi.schemas import MODE_LSI, MODE_VSM, Document, IndexParams, SearchHit, TfidfState


class Retriever:
    def __init__(self, documents: list[Document], vectorizer: TfidfVectorizer,
                 matrix: np.ndarray, lsi: LSIModel, params: IndexParams):
        self.documents = documents
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.lsi = lsi
        self.params = params
        self.doc_ids = [d.doc_id for d in documents]
        self.categories = np.array([d.category for d in documents])

    @classmethod
    def build(cls, documents: list[Document], params: IndexParams) -> "Retriever":
        tokens = [preprocess(d.narrative, params.use_stemming) for d in documents]
        vectorizer = TfidfVectorizer(params.tf_mode, params.min_df)
        state = vectorizer.fit(tokens)
        return cls(documents, vectorizer, state.matrix, LSIModel.fit(state.matrix), params)

    @classmethod
    def load_or_build(cls, documents: list[Document], params: IndexParams,
                      cache_path: Path = config.CACHE_PATH,
                      force: bool = False) -> tuple["Retriever", bool]:
        fingerprint = compute_fingerprint(documents, params)
        cached = None if force else load_cache(cache_path, fingerprint)
        if cached is not None:
            state, factors = cached
            vectorizer = TfidfVectorizer.from_state(state, params.tf_mode, params.min_df)
            return cls(documents, vectorizer, state.matrix, LSIModel(factors), params), True
        retriever = cls.build(documents, params)
        state = TfidfState(retriever.vectorizer.vocab, retriever.vectorizer.idf, retriever.matrix)
        save_cache(cache_path, state, retriever.lsi.factors, retriever.doc_ids, fingerprint)
        return retriever, False

    def query_vector(self, query: str) -> np.ndarray:
        return self.vectorizer.transform_query(preprocess(query, self.params.use_stemming))

    def scores(self, query: str, mode: str, k: int | None = None) -> np.ndarray:
        q = self.query_vector(query)
        if mode == MODE_VSM:
            return cosine_to_rows(q, self.matrix.T)
        if mode == MODE_LSI:
            if k is None:
                raise ValueError("Mode LSI membutuhkan nilai k")
            q_hat = self.lsi.project_query(q, k)
            return cosine_to_rows(q_hat, self.lsi.document_vectors(k))
        raise ValueError(f"Mode tidak dikenal: {mode}")

    def search(self, query: str, mode: str, k: int | None = None,
               top_n: int = config.DEFAULT_TOP_N) -> list[SearchHit]:
        scores = self.scores(query, mode, k)
        order = np.argsort(-scores, kind="stable")[:top_n]
        return [
            SearchHit(rank=i + 1, doc_id=self.doc_ids[j],
                      category=self.documents[j].category, score=round(float(scores[j]), 4))
            for i, j in enumerate(order)
        ]

    @staticmethod
    def hits_to_frame(hits: list[SearchHit]) -> pd.DataFrame:
        return pd.DataFrame([h.__dict__ for h in hits])

    def compare(self, query: str, k: int, top_n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        vsm = self.hits_to_frame(self.search(query, MODE_VSM, top_n=top_n)).add_prefix("VSM_")
        lsi = self.hits_to_frame(self.search(query, MODE_LSI, k=k, top_n=top_n)).add_prefix("LSI_")
        return pd.concat([vsm, lsi], axis=1)
