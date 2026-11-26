import numpy as np
import pandas as pd

from lsi.engine.retriever import Retriever
from lsi.schemas import MODE_LSI, MODE_VSM, EvalQuery


def relevance_in_rank_order(scores: np.ndarray, categories: np.ndarray, target: str) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    return (categories[order] == target).astype(int)


def precision_at_n(relevance: np.ndarray, n: int) -> float:
    return float(relevance[:n].sum() / n)


def average_precision(relevance: np.ndarray) -> float:
    if relevance.sum() == 0:
        return 0.0
    hits = np.cumsum(relevance)
    positions = np.arange(1, len(relevance) + 1)
    return float((hits / positions)[relevance == 1].mean())


def reciprocal_rank(relevance: np.ndarray) -> float:
    first = np.flatnonzero(relevance)
    return 0.0 if first.size == 0 else float(1.0 / (first[0] + 1))


def _row(query: EvalQuery, method: str, k: object, relevance: np.ndarray, top_n: int) -> dict:
    return {
        "query": query.query,
        "target": query.target_category,
        "method": method,
        "k": k,
        f"P@{top_n}": precision_at_n(relevance, top_n),
        "AP": average_precision(relevance),
        "RR": reciprocal_rank(relevance),
    }


def evaluate(retriever: Retriever, queries: list[EvalQuery],
             ks: list[int], top_n: int) -> pd.DataFrame:
    valid_ks = [k for k in ks if k <= retriever.lsi.rank]
    rows = []
    for query in queries:
        vsm = retriever.scores(query.query, MODE_VSM)
        rel = relevance_in_rank_order(vsm, retriever.categories, query.target_category)
        rows.append(_row(query, "VSM", "-", rel, top_n))
        for k in valid_ks:
            lsi = retriever.scores(query.query, MODE_LSI, k)
            rel = relevance_in_rank_order(lsi, retriever.categories, query.target_category)
            rows.append(_row(query, "LSI", k, rel, top_n))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    grouped = results.groupby(["method", "k"], sort=False)[[f"P@{top_n}", "AP", "RR"]].mean()
    grouped = grouped.rename(columns={"AP": "MAP", "RR": "MRR"})
    return grouped.round(4).reset_index()


def sensitivity(retriever: Retriever, summary: pd.DataFrame) -> pd.DataFrame:
    lsi = summary[summary["method"] == "LSI"].copy()
    lsi["k"] = lsi["k"].astype(int)
    lsi.insert(2, "energi_persen", [round(retriever.lsi.energy(k), 2) for k in lsi["k"]])
    return lsi.drop(columns=["method"]).reset_index(drop=True)
