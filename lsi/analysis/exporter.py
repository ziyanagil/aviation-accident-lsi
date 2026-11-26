from datetime import datetime
from pathlib import Path

import pandas as pd

from lsi import config

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
SUPPORTED_FORMATS = ("csv", "md", "tex")


def _format_cell(value: object, float_format: str) -> str:
    return float_format.format(value) if isinstance(value, float) else str(value)


def to_markdown(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_format_cell(v, float_format) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def escape_latex(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def to_latex(frame: pd.DataFrame, caption: str = "", label: str = "",
             float_format: str = "{:.4f}") -> str:
    alignment = "l" * len(frame.columns)
    header = " & ".join(escape_latex(str(c)) for c in frame.columns) + r" \\"
    rows = [
        " & ".join(escape_latex(_format_cell(v, float_format)) for v in row) + r" \\"
        for row in frame.itertuples(index=False)
    ]
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [rf"\begin{{tabular}}{{{alignment}}}", r"\hline", header, r"\hline", *rows,
              r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def export_table(frame: pd.DataFrame, name: str, formats: tuple[str, ...] = SUPPORTED_FORMATS,
                 directory: Path = config.TABLES_DIR, caption: str = "") -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Format tidak didukung: {fmt}")
        path = directory / f"{name}.{fmt}"
        if fmt == "csv":
            frame.to_csv(path, index=False)
        elif fmt == "md":
            path.write_text(to_markdown(frame) + "\n", encoding="utf-8")
        else:
            path.write_text(to_latex(frame, caption=caption, label=f"tab:{name}") + "\n",
                            encoding="utf-8")
        written.append(path)
    return written


def write_log(name: str, lines: list[str], directory: Path = config.LOGS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.log"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text("\n".join([f"[{stamp}] {name}", *lines]) + "\n", encoding="utf-8")
    return path
