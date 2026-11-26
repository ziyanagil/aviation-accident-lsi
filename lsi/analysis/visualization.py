from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsi.engine.retriever import Retriever
from lsi.schemas import MODE_LSI, MODE_VSM


def _finalize(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _mark_k(ax, k: int | None) -> None:
    if k is not None:
        ax.axvline(k, linestyle="--", color="red", label=f"k = {k}")
        ax.legend()


def scree_plot(singular_values: np.ndarray, path: Path, k: int | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(singular_values) + 1), singular_values, "o-")
    _mark_k(ax, k)
    ax.set(xlabel="Indeks i", ylabel=r"Nilai singular $\sigma_i$", title="Scree Plot")
    ax.grid(alpha=0.3)
    return _finalize(fig, path)


def energy_plot(cumulative_energy: np.ndarray, path: Path, k: int | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cumulative_energy) + 1), cumulative_energy, "s-")
    _mark_k(ax, k)
    ax.set(xlabel="k", ylabel="Energi kumulatif (%)", title="Energi Singular yang Dipertahankan")
    ax.grid(alpha=0.3)
    return _finalize(fig, path)


def map_vs_k_plot(summary: pd.DataFrame, path: Path) -> Path:
    lsi = summary[summary["method"] == "LSI"]
    baseline = summary[summary["method"] == "VSM"]["MAP"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lsi["k"].astype(int), lsi["MAP"], "o-", label="SVD-LSI")
    ax.axhline(baseline, linestyle="--", color="gray", label="VSM baseline")
    ax.set(xlabel="Dimensi k", ylabel="MAP", title="MAP terhadap Dimensi k")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finalize(fig, path)


def score_comparison_plot(retriever: Retriever, query: str, k: int, path: Path,
                          top_n: int = 5) -> Path:
    vsm = retriever.scores(query, MODE_VSM)
    lsi = retriever.scores(query, MODE_LSI, k)
    chosen = np.argsort(-lsi, kind="stable")[:top_n]
    positions = np.arange(len(chosen))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(positions - 0.2, vsm[chosen], width=0.4, label="VSM")
    ax.bar(positions + 0.2, lsi[chosen], width=0.4, label=f"SVD-LSI (k={k})")
    ax.set_xticks(positions)
    ax.set_xticklabels([retriever.doc_ids[j] for j in chosen], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="Cosine similarity", title=f'Perbandingan skor: "{query}"')
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _finalize(fig, path)


def score_vs_k_plot(retriever: Retriever, query: str, ks: list[int], path: Path,
                    top_docs: int = 5) -> Path:
    valid_ks = [k for k in ks if k <= retriever.lsi.rank]
    per_k = {k: retriever.scores(query, MODE_LSI, k) for k in valid_ks}
    chosen = np.argsort(-per_k[max(valid_ks)], kind="stable")[:top_docs]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j in chosen:
        ax.plot(valid_ks, [per_k[k][j] for k in valid_ks], "o-", label=retriever.doc_ids[j])
    ax.set(xlabel="Dimensi k", ylabel="Cosine similarity", title=f'Skor terhadap k: "{query}"')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _finalize(fig, path)
