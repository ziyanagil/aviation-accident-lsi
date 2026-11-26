import re
import unicodedata

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"aviation safety network", re.IGNORECASE),
    re.compile(r"flight\s*safety\s*foundation", re.IGNORECASE),
    re.compile(r"flightsafety\.org", re.IGNORECASE),
    re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}", re.IGNORECASE),
    re.compile(r"copyright|all rights reserved|©", re.IGNORECASE),
    re.compile(r"^\s*(share|tweet|print|follow us|cookie|privacy policy)\b", re.IGNORECASE),
]

URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
NUMERIC_TOKEN_RE = re.compile(r"\b\d[\d.,:/\-]*\b")
SPECIAL_CHAR_RE = re.compile(r"[^A-Za-z0-9\s.,;:'\-]")
WHITESPACE_RE = re.compile(r"\s+")

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})")
TYPE_RE = re.compile(
    r"Type:\s*(.+?)\s+(?:Owner/operator:|Registration:|MSN:|Year of manufacture:|"
    r"Total airframe|Engines:|Crew:|Fatalities:)", re.IGNORECASE)
PHASE_RE = re.compile(
    r"Phase:\s*(.+?)(?:\s*\(|\s+Departure airport:|\s+Nature:|\s+Narrative:|\s+Destination)",
    re.IGNORECASE)
NARRATIVE_RE = re.compile(
    r"Narrative:\s*(.+?)(?:\s+Sources?:|\s+Follow-up|\s+Accident investigation:|"
    r"\s+Photos?:|\s+Classification:|$)", re.IGNORECASE | re.DOTALL)
SOURCE_URL_RE = re.compile(r"https?://asn\.flightsafety\.org/\S+", re.IGNORECASE)

CATEGORY_KEYWORDS = {
    "ENGINE": ["engine", "powerplant", "turbine", "compressor", "thrust", "uncontained",
               "flameout", "flamed out", "propeller"],
    "RUNWAY": ["runway", "overran", "overrun", "excursion", "veered", "tarmac",
               "landing gear", "undercarriage", "skidded", "rollout"],
    "WEATHER": ["windshear", "wind shear", "microburst", "icing", "turbulence",
                "visibility", "fog", "thunderstorm", "downdraft", "ice"],
    "CFIT": ["controlled flight into terrain", "terrain", "mountain", "ridge", "hillside",
             "altitude deviation", "glidepath", "waypoint", "approach chart"],
}
UNLABELED = "UNLABELED"


def is_boilerplate(line: str) -> bool:
    return any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [line for line in text.splitlines() if line.strip() and not is_boilerplate(line)]
    text = " ".join(lines)
    text = URL_RE.sub(" ", text)
    text = NUMERIC_TOKEN_RE.sub(" ", text)
    text = SPECIAL_CHAR_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_date(raw: str) -> str:
    match = DATE_RE.search(raw)
    if not match:
        return ""
    day, month, year = match.groups()
    month_number = MONTHS.get(month[:3].lower())
    if not month_number:
        return ""
    return f"{int(year):04d}-{month_number:02d}-{int(day):02d}"


def parse_asn_text(text: str) -> dict[str, str]:
    flat = WHITESPACE_RE.sub(" ", text)
    date_match = re.search(r"Date:\s*(.+?)\s+(?:Time:|Type:)", flat, re.IGNORECASE)
    type_match = TYPE_RE.search(flat)
    phase_match = PHASE_RE.search(flat)
    narrative_match = NARRATIVE_RE.search(flat)
    url_match = SOURCE_URL_RE.search(flat)
    return {
        "date": parse_date(date_match.group(1)) if date_match else "",
        "aircraft_type": type_match.group(1).strip() if type_match else "",
        "phase": phase_match.group(1).strip() if phase_match else "",
        "narrative": narrative_match.group(1).strip() if narrative_match else "",
        "source_url": url_match.group(0).rstrip(".,;)") if url_match else "",
    }


def infer_category(narrative: str) -> str:
    lowered = narrative.lower()
    scores = {
        category: sum(lowered.count(keyword) for keyword in keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else UNLABELED
