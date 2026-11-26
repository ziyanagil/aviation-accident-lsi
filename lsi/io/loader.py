import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lsi.schemas import Document, EvalQuery, IndexParams, SvdFactors, TfidfState

COLUMNS = ["doc_id", "date", "aircraft_type", "phase", "category", "narrative", "source_url"]


def load_documents(path: Path) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    if path.suffix.lower() == ".json":
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    else:
        frame = pd.read_csv(path)
    missing = [c for c in ("doc_id", "narrative") if c not in frame.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[COLUMNS].fillna("").astype(str)
    frame = frame[frame["narrative"].str.strip() != ""].reset_index(drop=True)
    return [Document(**row) for row in frame.to_dict(orient="records")]


def save_documents(documents: list[Document], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{column: getattr(doc, column) for column in COLUMNS} for doc in documents]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def load_queries(path: Path) -> list[EvalQuery]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuery(**item) for item in raw]


def compute_fingerprint(documents: list[Document], params: IndexParams) -> str:
    digest = hashlib.sha1()
    digest.update(repr(params).encode("utf-8"))
    for doc in documents:
        digest.update(doc.doc_id.encode("utf-8"))
        digest.update(doc.narrative.encode("utf-8"))
    return digest.hexdigest()


def save_cache(path: Path, tfidf: TfidfState, factors: SvdFactors,
               doc_ids: list[str], fingerprint: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        matrix=tfidf.matrix,
        idf=tfidf.idf,
        vocab=np.array(tfidf.vocab, dtype=str),
        U=factors.U,
        S=factors.S,
        Vt=factors.Vt,
        doc_ids=np.array(doc_ids, dtype=str),
        fingerprint=np.array(fingerprint, dtype=str),
    )
    return path


def load_cache(path: Path, fingerprint: str) -> tuple[TfidfState, SvdFactors] | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            return None
        tfidf = TfidfState(
            vocab=[str(t) for t in data["vocab"]],
            idf=data["idf"],
            matrix=data["matrix"],
        )
        factors = SvdFactors(U=data["U"], S=data["S"], Vt=data["Vt"])
    return tfidf, factors
