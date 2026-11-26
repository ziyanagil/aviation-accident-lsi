from dataclasses import dataclass
from typing import Literal

import numpy as np

SearchMode = Literal["vsm", "lsi"]
MODE_VSM: SearchMode = "vsm"
MODE_LSI: SearchMode = "lsi"


@dataclass(frozen=True)
class Document:
    doc_id: str
    narrative: str
    date: str = ""
    aircraft_type: str = ""
    phase: str = ""
    category: str = ""
    source_url: str = ""


@dataclass(frozen=True)
class SearchHit:
    rank: int
    doc_id: str
    category: str
    score: float


@dataclass(frozen=True)
class EvalQuery:
    query: str
    target_category: str
    note: str = ""


@dataclass
class TfidfState:
    vocab: list[str]
    idf: np.ndarray
    matrix: np.ndarray


@dataclass
class SvdFactors:
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray


@dataclass(frozen=True)
class IndexParams:
    tf_mode: str
    min_df: int
    use_stemming: bool
