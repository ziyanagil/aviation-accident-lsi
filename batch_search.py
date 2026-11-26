from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

from lsi import config
from lsi.analysis import exporter
from lsi.engine.retriever import Retriever
from lsi.io.loader import load_documents
from lsi.schemas import MODE_LSI, IndexParams

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_FILE = config.EVAL_DIR / "queries_100.json"
TABLES_DIR = config.TABLES_DIR
BATCH_OUT_DIR = TABLES_DIR / "batch"


def slugify(text: str, max_len: int = 45) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "query"


def load_queries_from_file(path: Path, category_filter: str | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File batch query tidak ditemukan: {path}")

    items: list[dict] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for idx, entry in enumerate(raw, start=1):
            if isinstance(entry, str):
                items.append({"id": idx, "query": entry.strip(), "target_category": "GENERAL"})
            elif isinstance(entry, dict) and entry.get("query"):
                items.append(
                    {
                        "id": entry.get("id", idx),
                        "query": str(entry["query"]).strip(),
                        "target_category": str(entry.get("target_category", "GENERAL")).strip(),
                    }
                )
    else:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            items.append({"id": idx, "query": clean, "target_category": "GENERAL"})

    if category_filter:
        target = category_filter.upper()
        items = [it for it in items if it["target_category"].upper() == target]

    return items


def build_retriever(args: argparse.Namespace) -> Retriever:
    config.ensure_dirs()
    data_path = args.data or config.DATASET_PATH
    cache_path = args.cache or config.CACHE_PATH
    tf_mode = args.tf or config.DEFAULT_TF_MODE
    min_df = args.min_df if args.min_df is not None else config.DEFAULT_MIN_DF
    params = IndexParams(
        tf_mode=tf_mode,
        min_df=min_df,
        use_stemming=not args.no_stem,
    )
    documents = load_documents(data_path)
    retriever, _ = Retriever.load_or_build(
        documents,
        params,
        cache_path,
        force=args.no_cache,
    )
    return retriever


def run_single_query(
    retriever: Retriever,
    query: str,
    idx: int,
    category: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[bool, str]:
    try:
        if args.mode == "both":
            frame = retriever.compare(query, args.k, args.top)
        else:
            k = args.k if args.mode == MODE_LSI else None
            frame = retriever.hits_to_frame(retriever.search(query, args.mode, k, args.top))
    except Exception as exc:
        err = str(exc)
        print(f"[ERR] Q{idx:03d} ({query}): {err}", file=sys.stderr)
        return False, err

    output_text = frame.to_string(index=False)
    print(f"\n=== [Q{idx:03d} | {category}] {query} ===")
    print(output_text)

    if args.export:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"q{idx:03d}_{category.lower()}_{slugify(query)}"
        exporter.export_table(frame, prefix, tuple(args.export), directory=out_dir)

    return True, output_text


def run_interactive(args: argparse.Namespace, retriever: Retriever) -> None:
    print("Mode pencarian interaktif (ketik 'exit' atau 'quit' untuk keluar).")
    print(f"Konfigurasi aktif: k={args.k}, top={args.top}, mode={args.mode}, export={args.export or 'off'}\n")
    idx = 1
    while True:
        try:
            query = input(f"[Q{idx:03d}] Masukkan query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar dari mode interaktif.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Selesai.")
            break

        run_single_query(retriever, query, idx, "CLI", args, args.out_dir)
        idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jalankan pencarian SVD-LSI secara batch (file) maupun langsung dari terminal."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--file",
        type=Path,
        help=f"File daftar query (.json atau .txt). Contoh: {DEFAULT_BATCH_FILE.relative_to(PROJECT_ROOT)}",
    )
    source.add_argument(
        "--queries",
        nargs="+",
        help="Satu atau beberapa query langsung dari terminal (dipisahkan spasi, bungkus tanda kutip).",
    )
    source.add_argument(
        "--interactive",
        action="store_true",
        help="Mode terminal interaktif (ketik query satu per satu berulang kali).",
    )

    parser.add_argument("--k", type=int, default=config.DEFAULT_K, help=f"Dimensi ruang laten SVD (default: {config.DEFAULT_K})")
    parser.add_argument("--top", type=int, default=config.DEFAULT_TOP_N, help=f"Jumlah dokumen teratas (default: {config.DEFAULT_TOP_N})")
    parser.add_argument(
        "--mode",
        choices=["both", "vsm", "lsi"],
        default="both",
        help="Mode pencarian (default: both)",
    )
    parser.add_argument(
        "--export",
        nargs="+",
        choices=["md", "csv", "tex"],
        default=["md", "csv", "tex"],
        help="Format ekspor per query (default: md csv tex). Gunakan --no-export untuk menonaktifkan.",
    )
    parser.add_argument("--no-export", action="store_true", help="Tampilkan di terminal saja tanpa ekspor file.")
    parser.add_argument("--category", type=str, default=None, help="Filter target_category saat mode --file JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah query yang dijalankan.")
    parser.add_argument("--out-dir", type=Path, default=BATCH_OUT_DIR, help="Folder arsip hasil batch.")
    parser.add_argument("--summary-name", type=str, default="rekap_semua_query.md", help="Nama file rekap gabungan Markdown.")

    # Opsi passthrough ke indeks
    parser.add_argument("--data", type=Path, default=None, help="Override path dataset.")
    parser.add_argument("--cache", type=Path, default=None, help="Override path file cache .npz.")
    parser.add_argument("--tf", choices=["raw", "log"], default=None, help="Override skema pembobotan TF.")
    parser.add_argument("--min-df", type=int, default=None, help="Override batas minimum frekuensi dokumen.")
    parser.add_argument("--no-stem", action="store_true", help="Nonaktifkan stemming.")
    parser.add_argument("--no-cache", action="store_true", help="Abaikan cache .npz dan bangun ulang indeks.")

    args = parser.parse_args()
    if args.no_export:
        args.export = []
    return args


def main() -> None:
    args = parse_args()

    if args.interactive:
        retriever = build_retriever(args)
        run_interactive(args, retriever)
        return

    if args.queries:
        items = [{"id": i, "query": q.strip(), "target_category": "CLI"} for i, q in enumerate(args.queries, start=1)]
    else:
        batch_file = args.file or DEFAULT_BATCH_FILE
        items = load_queries_from_file(batch_file, category_filter=args.category)

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        print("Tidak ada query yang cocok untuk dijalankan.")
        return

    start = time.time()
    retriever = build_retriever(args)
    summary_sections: list[str] = [
        f"# Rekap Hasil Pencarian Batch (k={args.k}, top={args.top}, mode={args.mode})\n"
    ]
    success_count = 0

    for item in items:
        ok, stdout = run_single_query(
            retriever=retriever,
            query=item["query"],
            idx=int(item["id"]),
            category=item["target_category"],
            args=args,
            out_dir=args.out_dir,
        )
        if ok:
            success_count += 1
            summary_sections.append(
                f"## Q{int(item['id']):03d} [{item['target_category']}] — `{item['query']}`\n\n```text\n{stdout}\n```\n"
            )

    if args.export:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        combined_md = args.out_dir / args.summary_name
        combined_md.write_text("\n".join(summary_sections), encoding="utf-8")
        print(f"\nArsip per-query tersimpan di: {args.out_dir}")
        print(f"Rekap gabungan tersimpan di : {combined_md}")

    elapsed = time.time() - start
    print(f"Selesai: {success_count}/{len(items)} query berhasil diproses dalam {elapsed:.1f} detik.")
    if success_count < len(items):
        sys.exit(1)


if __name__ == "__main__":
    main()