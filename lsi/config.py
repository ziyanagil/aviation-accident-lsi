from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

DATASET_PATH = PROCESSED_DIR / "dataset_asn.csv"
CACHE_PATH = PROCESSED_DIR / "index_cache.npz"
QUERIES_PATH = EVAL_DIR / "queries.json"

DEFAULT_K = 150
DEFAULT_TOP_N = 5
DEFAULT_MIN_DF = 1
DEFAULT_TF_MODE = "log"
DEFAULT_USE_STEMMING = True
K_SWEEP = [4, 10, 25, 50, 100, 150, 200, 300, 500]
MIN_NARRATIVE_WORDS = 8
SVD_RANK_TOLERANCE = 1e-10

ASN_BASE_URL = "https://asn.flightsafety.org"
YEARS = list(range(2000, 2027))
DOWNLOAD_DELAY_SECONDS = 1.5
USER_AGENT = "asn-lsi-research/1.0 (academic coursework)"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, EVAL_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
