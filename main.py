import argparse
import logging
import sys
from pathlib import Path

from lsi import config
from lsi.analysis import evaluation, exporter, visualization
from lsi.engine.retriever import Retriever
from lsi.io.loader import load_documents, load_queries, save_documents
from lsi.io.pdf_reader import ingest_pdfs
from lsi.schemas import MODE_LSI, MODE_VSM, IndexParams


def add_index_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=config.DATASET_PATH)
    parser.add_argument("--cache", type=Path, default=config.CACHE_PATH)
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--top", type=int, default=config.DEFAULT_TOP_N)
    parser.add_argument("--tf", choices=["raw", "log"], default=config.DEFAULT_TF_MODE)
    parser.add_argument("--min-df", type=int, default=config.DEFAULT_MIN_DF)
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--no-cache", action="store_true")


def index_params(args: argparse.Namespace) -> IndexParams:
    return IndexParams(tf_mode=args.tf, min_df=args.min_df, use_stemming=not args.no_stem)


def output_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base = Path(args.output_dir)
    return base / "tables", base / "figures", base / "logs"


def make_retriever(args: argparse.Namespace, force: bool = False) -> tuple[Retriever, bool]:
    documents = load_documents(args.data)
    return Retriever.load_or_build(
        documents, index_params(args), args.cache, force=force or args.no_cache)


def command_ingest(args: argparse.Namespace) -> None:
    documents = ingest_pdfs(args.raw_dir, min_words=args.min_words)
    if not documents:
        raise SystemExit("Tidak ada dokumen yang berhasil diekstraksi")
    save_documents(documents, args.out)
    print(f"{len(documents)} dokumen ditulis ke {args.out}")


def command_build(args: argparse.Namespace) -> None:
    _, figures_dir, _ = output_dirs(args)
    retriever, _ = make_retriever(args, force=True)
    k = min(args.k, retriever.lsi.rank)
    for key, value in retriever.lsi.dimension_report(k).items():
        print(f"{key:16s}: {value}")
    left, right = retriever.lsi.orthogonality_error()
    print(f"{'ortogonalitas':16s}: U={left:.2e}, V={right:.2e}")
    visualization.scree_plot(retriever.lsi.singular_values, figures_dir / "scree_plot.png", k)
    visualization.energy_plot(
        retriever.lsi.cumulative_energy(), figures_dir / "energy_plot.png", k)
    print(f"cache: {args.cache}")


def command_search(args: argparse.Namespace) -> None:
    tables_dir, _, _ = output_dirs(args)
    retriever, _ = make_retriever(args)
    if args.mode == "both":
        frame = retriever.compare(args.query, args.k, args.top)
    else:
        k = args.k if args.mode == MODE_LSI else None
        frame = retriever.hits_to_frame(retriever.search(args.query, args.mode, k, args.top))
    print(frame.to_string(index=False))
    if args.export:
        exporter.export_table(frame, args.out_name, tuple(args.export), directory=tables_dir)


def command_eval(args: argparse.Namespace) -> None:
    tables_dir, figures_dir, logs_dir = output_dirs(args)
    tables_dir.mkdir(parents=True, exist_ok=True)
    retriever, _ = make_retriever(args)
    queries = load_queries(args.queries)
    k_sweep = args.k_sweep
    results = evaluation.evaluate(retriever, queries, k_sweep, args.top)
    summary = evaluation.summarize(results, args.top)
    sensitivity = evaluation.sensitivity(retriever, summary)

    exporter.export_table(results, "metrik_per_query", ("csv",), directory=tables_dir)
    exporter.export_table(summary, "metrik_ringkasan", directory=tables_dir, caption="Ringkasan metrik evaluasi")
    exporter.export_table(sensitivity, "sensitivitas_k", directory=tables_dir, caption="Sensitivitas terhadap dimensi k")

    sections = []
    for item in queries:
        frame = retriever.compare(item.query, args.k, args.top)
        sections.append(f'### Query: "{item.query}" (target {item.target_category}, k={args.k})\n')
        sections.append(exporter.to_markdown(frame) + "\n")
    (tables_dir / "ranking_per_query.md").write_text("\n".join(sections), encoding="utf-8")

    visualization.scree_plot(retriever.lsi.singular_values, figures_dir / "scree_plot.png", args.k)
    visualization.energy_plot(
        retriever.lsi.cumulative_energy(), figures_dir / "energy_plot.png", args.k)
    visualization.map_vs_k_plot(summary, figures_dir / "map_vs_k.png")
    for index, item in enumerate(queries, start=1):
        visualization.score_comparison_plot(
            retriever, item.query, args.k, figures_dir / f"skor_q{index}.png", args.top)
    visualization.score_vs_k_plot(
        retriever, queries[0].query, k_sweep, figures_dir / "skor_vs_k.png")

    report = retriever.lsi.dimension_report(min(args.k, retriever.lsi.rank))
    log_lines = [f"{key}: {value}" for key, value in report.items()]
    log_lines += ["", exporter.to_markdown(summary)]
    log_path = exporter.write_log(args.log_name, log_lines, directory=logs_dir)
    print(summary.to_string(index=False))
    print(f"\ntabel: {tables_dir} | grafik: {figures_dir} | log: {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistem temu balik informasi SVD-LSI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="PDF di data/raw -> dataset CSV")
    ingest.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ingest.add_argument("--out", type=Path, default=config.DATASET_PATH)
    ingest.add_argument("--min-words", type=int, default=config.MIN_NARRATIVE_WORDS)
    ingest.set_defaults(handler=command_ingest)

    build = sub.add_parser("build", help="bangun indeks TF-IDF dan SVD, simpan cache")
    add_index_options(build)
    build.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    build.set_defaults(handler=command_build)

    search = sub.add_parser("search", help="cari dokumen")
    add_index_options(search)
    search.add_argument("query")
    search.add_argument("--mode", choices=["both", MODE_VSM, MODE_LSI], default="both")
    search.add_argument("--export", nargs="+", choices=list(exporter.SUPPORTED_FORMATS))
    search.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    search.add_argument("--out-name", type=str, default="hasil_pencarian")
    search.set_defaults(handler=command_search)

    evaluate = sub.add_parser("eval", help="evaluasi query uji dan sensitivitas k")
    add_index_options(evaluate)
    evaluate.add_argument("--queries", type=Path, default=config.QUERIES_PATH)
    evaluate.add_argument("--k-sweep", type=int, nargs="+", default=list(config.K_SWEEP))
    evaluate.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    evaluate.add_argument("--log-name", type=str, default="eval_summary")
    evaluate.set_defaults(handler=command_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.ensure_dirs()
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
