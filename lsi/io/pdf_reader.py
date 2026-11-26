import logging
import re
from pathlib import Path

from pypdf import PdfReader

from lsi import config
from lsi.nlp.cleaner import clean_text, infer_category, parse_asn_text
from lsi.schemas import Document

logger = logging.getLogger(__name__)
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9]+")


def extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def list_pdfs(raw_dir: Path) -> list[Path]:
    return sorted(Path(raw_dir).rglob("*.pdf"))


def build_doc_id(path: Path) -> str:
    year = path.parent.name
    stem = SAFE_ID_RE.sub("-", path.stem).strip("-")
    return f"ASN-{year}-{stem}"


def pdf_to_document(path: Path, min_words: int = config.MIN_NARRATIVE_WORDS) -> Document | None:
    raw_text = extract_text(path)
    parsed = parse_asn_text(raw_text)
    narrative = clean_text(parsed["narrative"] or raw_text)
    if len(narrative.split()) < min_words:
        logger.warning("Narasi terlalu pendek, dilewati: %s", path.name)
        return None
    return Document(
        doc_id=build_doc_id(path),
        narrative=narrative,
        date=parsed["date"],
        aircraft_type=parsed["aircraft_type"],
        phase=parsed["phase"],
        category=infer_category(narrative),
        source_url=parsed["source_url"],
    )


def ingest_pdfs(raw_dir: Path, min_words: int = config.MIN_NARRATIVE_WORDS) -> list[Document]:
    documents: list[Document] = []
    for path in list_pdfs(raw_dir):
        try:
            document = pdf_to_document(path, min_words=min_words)
        except Exception as error:
            logger.error("Gagal membaca %s: %s", path, error)
            continue
        if document is not None:
            documents.append(document)
    return documents
