"""Command line entry point.

    python -m fitpipeline DATA_DIR [OUTPUT.json] [-v] [--mesh mesh_02 ...] [--segment 0 ...] [--no-write]

Classifies and fits every segment under DATA_DIR, prints a table, and writes
OUTPUT.json (default `predictions.json`). `--mesh`/`--segment` only filter the
printed table; the written file always covers the whole dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from fitpipeline.loader import iter_mesh_ids
from fitpipeline.predictions import predict_dataset, save_predictions
from fitpipeline.report import format_report, format_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fitpipeline",
        description="Classify every mesh segment, fit its primitive, and write predictions.json.",
    )
    parser.add_argument("data_dir", help="directory containing mesh_XX/ folders")
    parser.add_argument(
        "output", nargs="?", default="predictions.json",
        help="path of the JSON file to write (default: predictions.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print each decision's reason and fitted parameters")
    parser.add_argument("--mesh", action="append", metavar="MESH_ID",
                        help="only show this mesh in the table (repeatable)")
    parser.add_argument("--segment", action="append", type=int, metavar="ID",
                        help="only show this segment id in the table (repeatable)")
    parser.add_argument("--no-write", action="store_true", help="print the table but do not write a file")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the final summary line")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(f"data directory not found: {data_dir}")
    available = iter_mesh_ids(data_dir)
    if not available:
        parser.error(f"no mesh_XX/mesh.json found under {data_dir}")
    unknown = sorted(set(args.mesh or []) - set(available))
    if unknown:
        parser.error(f"unknown mesh id(s) {', '.join(unknown)}; available: {', '.join(available)}")

    start = time.perf_counter()
    predictions = predict_dataset(data_dir)
    elapsed = time.perf_counter() - start

    shown = [
        p for p in predictions
        if (not args.mesh or p["mesh_id"] in args.mesh)
        and (not args.segment or p["segment_id"] in args.segment)
    ]
    if not args.no_write:
        save_predictions(predictions, args.output)

    if not args.quiet:
        color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        print(format_report(shown, verbose=args.verbose, color=color))
        print()
    written = "not written" if args.no_write else f"-> {args.output}"
    print(f"{format_summary(predictions)} {written} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
