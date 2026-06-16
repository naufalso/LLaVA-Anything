#!/usr/bin/env python3
"""Aggregate VLMEvalKit benchmark score CSV files into a paper-ready wide CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = Path("output/vlmevalkit")
DEFAULT_OUTPUT = Path("output/reports/vlmevalkit_summary.csv")
DEFAULT_RAW_OUTPUT = Path("output/reports/vlmevalkit_raw_summary.csv")
DEFAULT_MM_OUTPUT = Path("output/reports/vlmevalkit_mm_summary.csv")


AGGREGATE_SUFFIXES = {
    "_MTL_MMBench_DEV_acc.csv": "MTL_MMBench_DEV",
    "_MMMB_acc.csv": "MMMB",
}

LANGUAGE_SUFFIXES = (
    ("_MMMB_", "_acc.csv", "MMMB"),
    ("_MMBench_dev_", "_acc.csv", "MTL_MMBench_DEV"),
)


def parse_aggregate_filename(path: Path) -> tuple[str, str] | None:
    name = path.name
    for suffix, benchmark in AGGREGATE_SUFFIXES.items():
        if name.endswith(suffix):
            model = name[: -len(suffix)]
            if model:
                return model, benchmark
    return None


def parse_language_filename(path: Path) -> tuple[str, str, str] | None:
    name = path.name
    for marker, suffix, benchmark in LANGUAGE_SUFFIXES:
        if marker not in name or not name.endswith(suffix):
            continue
        prefix, remainder = name.rsplit(marker, 1)
        language = remainder[: -len(suffix)]
        if not prefix or not language:
            continue
        dataset = f"MMMB_{language}" if benchmark == "MMMB" else f"MMBench_dev_{language}"
        return prefix, benchmark, dataset
    return None


def numeric(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def mean_overall(path: Path) -> float | None:
    values: list[float] = []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = numeric(row.get("Overall"))
                if score is not None:
                    values.append(score)
    except OSError:
        return None

    if not values:
        return None
    return sum(values) / len(values)


def sanitize_column(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def dataset_column_base(benchmark: str, dataset: str) -> str:
    dataset = dataset.strip()
    if benchmark == "MMMB" and dataset.startswith("MMMB_"):
        return f"mmmb_{sanitize_column(dataset.removeprefix('MMMB_'))}"
    if benchmark == "MTL_MMBench_DEV" and dataset.startswith("MMBench_dev_"):
        return f"mtl_mmbench_dev_{sanitize_column(dataset.removeprefix('MMBench_dev_'))}"
    return f"{sanitize_column(benchmark)}_{sanitize_column(dataset)}"


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def companion_raw_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_raw_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_raw_summary{output_csv.suffix}")


def companion_mm_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_mm_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_mm_summary{output_csv.suffix}")


def aggregate_vlmevalkit_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
) -> Path:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for score_path in input_dir.rglob("*_acc.csv"):
        parsed = parse_aggregate_filename(score_path)
        if parsed is None:
            continue
        model, benchmark = parsed
        score = mean_overall(score_path)
        if score is None:
            continue
        mtime = score_path.stat().st_mtime
        key = (model, benchmark)
        if key not in latest or mtime >= latest[key][0]:
            latest[key] = (mtime, score)

    by_model: dict[str, dict[str, float]] = {}
    benchmarks: set[str] = set()
    for (model, benchmark), (_mtime, score) in latest.items():
        by_model.setdefault(model, {})[benchmark] = score
        benchmarks.add(benchmark)

    ordered_benchmarks = sorted(benchmarks)
    rows = []
    for model, scores in by_model.items():
        values = [scores[name] for name in ordered_benchmarks if name in scores]
        average = sum(values) / len(values) if values else 0.0
        rows.append((average, model, scores))
    rows.sort(key=lambda item: (item[0], item[1]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *ordered_benchmarks, "average"])
        for average, model, scores in rows:
            writer.writerow(
                [
                    model,
                    *(format_float(scores.get(name)) for name in ordered_benchmarks),
                    format_float(average),
                ]
            )

    return output_csv


def iter_language_overalls(path: Path, benchmark: str, default_dataset: str | None = None):
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("DATASET") or default_dataset
                if not dataset:
                    continue
                score = numeric(row.get("Overall"))
                if score is None:
                    continue
                yield dataset_column_base(benchmark, dataset), score
    except OSError:
        return


def iter_raw_scores(path: Path, benchmark: str, default_dataset: str | None = None):
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("DATASET") or default_dataset
                if not dataset:
                    continue
                base = dataset_column_base(benchmark, dataset)
                for header, value in row.items():
                    if header in (None, "", "split", "DATASET"):
                        continue
                    score = numeric(value)
                    if score is None:
                        continue
                    metric = sanitize_column(header)
                    column = base if metric == "overall" else f"{base}_{metric}"
                    yield column, score
    except OSError:
        return


def aggregate_vlmevalkit_raw_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_RAW_OUTPUT,
) -> Path:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for score_path in input_dir.rglob("*_acc.csv"):
        aggregate = parse_aggregate_filename(score_path)
        language = parse_language_filename(score_path)

        if aggregate is not None:
            model, benchmark = aggregate
            default_dataset = None
        elif language is not None:
            model, benchmark, default_dataset = language
        else:
            continue

        mtime = score_path.stat().st_mtime
        for column, score in iter_raw_scores(score_path, benchmark, default_dataset):
            key = (model, column)
            if key not in latest or mtime >= latest[key][0]:
                latest[key] = (mtime, score)

    by_model: dict[str, dict[str, float]] = {}
    columns: set[str] = set()
    for (model, column), (_mtime, score) in latest.items():
        by_model.setdefault(model, {})[column] = score
        columns.add(column)

    ordered_columns = sorted(columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *ordered_columns])
        for model in sorted(by_model):
            scores = by_model[model]
            writer.writerow([model, *(format_float(scores.get(name)) for name in ordered_columns)])

    return output_csv


def aggregate_vlmevalkit_mm_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_MM_OUTPUT,
) -> Path:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for score_path in input_dir.rglob("*_acc.csv"):
        aggregate = parse_aggregate_filename(score_path)
        language = parse_language_filename(score_path)

        if aggregate is not None:
            model, benchmark = aggregate
            default_dataset = None
        elif language is not None:
            model, benchmark, default_dataset = language
        else:
            continue

        mtime = score_path.stat().st_mtime
        for column, score in iter_language_overalls(score_path, benchmark, default_dataset):
            key = (model, column)
            if key not in latest or mtime >= latest[key][0]:
                latest[key] = (mtime, score)

    by_model: dict[str, dict[str, float]] = {}
    columns: set[str] = set()
    for (model, column), (_mtime, score) in latest.items():
        by_model.setdefault(model, {})[column] = score
        columns.add(column)

    ordered_columns = sorted(columns)
    rows = []
    for model, scores in by_model.items():
        values = [scores[name] for name in ordered_columns if name in scores]
        average = sum(values) / len(values) if values else 0.0
        rows.append((average, model, scores))
    rows.sort(key=lambda item: (item[0], item[1]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *ordered_columns, "average"])
        for average, model, scores in rows:
            writer.writerow(
                [
                    model,
                    *(format_float(scores.get(name)) for name in ordered_columns),
                    format_float(average),
                ]
            )

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help=(
            "Path for the expanded raw metrics CSV. Defaults to a companion "
            "*_raw_summary.csv next to --output."
        ),
    )
    parser.add_argument(
        "--mm-output",
        type=Path,
        default=None,
        help=(
            "Path for the language-level overall metrics CSV. Defaults to a companion "
            "*_mm_summary.csv next to --output."
        ),
    )
    parser.add_argument(
        "--no-raw-summary",
        action="store_true",
        help="Only write the benchmark-level summary CSV.",
    )
    parser.add_argument(
        "--no-mm-summary",
        action="store_true",
        help="Do not write the language-level overall metrics CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = aggregate_vlmevalkit_results(args.input, args.output)
    print(output)
    if not args.no_raw_summary:
        raw_output = args.raw_output or companion_raw_output(output)
        raw_output = aggregate_vlmevalkit_raw_results(args.input, raw_output)
        print(raw_output)
    if not args.no_mm_summary:
        mm_output = args.mm_output or companion_mm_output(output)
        mm_output = aggregate_vlmevalkit_mm_results(args.input, mm_output)
        print(mm_output)


if __name__ == "__main__":
    main()
