#!/usr/bin/env python3
"""Aggregate lm-evaluation-harness result JSON files into paper-ready CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("output/lm-eval/multilingual-lm-eval")
DEFAULT_OUTPUT = Path("output/reports/lm_eval_multilingual_summary.csv")
DEFAULT_LONG_OUTPUT = Path("output/reports/lm_eval_multilingual_long_summary.csv")
DEFAULT_RAW_OUTPUT = Path("output/reports/lm_eval_multilingual_raw_summary.csv")

DEFAULT_BENCHMARK_ORDER = (
    "agieval",
    "arc_multilingual",
    "blend_sample",
    "cultural_bench",
    "global_mmlu_gen_0shot",
    "hellaswag_multilingual",
    "include_base_44_gen_0shot",
    "multi_if",
    "truthfulqa_multilingual_mc2",
)

PRIMARY_METRICS_BY_BENCHMARK = {
    "agieval": ("acc,none",),
    "arc_multilingual": ("acc_norm,none", "acc,none"),
    "blend_sample": ("acc_norm,none", "acc,none"),
    "cultural_bench": ("acc_norm,none", "acc,none"),
    "global_mmlu_gen_0shot": ("exact_match,extract-answer", "exact_match,none"),
    "hellaswag_multilingual": ("acc_norm,none", "acc,none"),
    "include_base_44_gen_0shot": ("exact_match,extract-answer", "exact_match,none"),
    "multi_if": ("prompt_level_strict_acc,none", "inst_level_strict_acc,none"),
    "truthfulqa_multilingual_mc2": ("acc,none",),
}

FALLBACK_PRIMARY_METRICS = (
    "exact_match,extract-answer",
    "exact_match,none",
    "acc_norm,none",
    "acc,none",
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def sanitize_column(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def companion_long_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_long_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_long_summary{output_csv.suffix}")


def companion_raw_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_raw_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_raw_summary{output_csv.suffix}")


def result_timestamp(path: Path) -> float:
    match = re.search(
        r"results_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2}(?:\.\d+)?)",
        path.name,
    )
    if match:
        raw = f"{match.group(1)}T{match.group(2)}:{match.group(3)}:{match.group(4)}"
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def model_and_benchmark(path: Path, input_dir: Path, payload: dict[str, Any]) -> tuple[str, str] | None:
    try:
        parts = path.relative_to(input_dir).parts
    except ValueError:
        parts = path.parts

    if len(parts) >= 3:
        return parts[0], parts[1]

    model_name = payload.get("model_name") or payload.get("model_name_sanitized")
    model = Path(str(model_name)).name if model_name else None
    config = payload.get("config")
    task = None
    if isinstance(config, dict):
        tasks = config.get("tasks")
        if isinstance(tasks, str):
            task = tasks
        elif isinstance(tasks, list) and tasks:
            task = str(tasks[0])
    if model and task:
        return model, task
    return None


def metric_candidates(benchmark: str) -> tuple[str, ...]:
    return PRIMARY_METRICS_BY_BENCHMARK.get(benchmark, FALLBACK_PRIMARY_METRICS)


def primary_score(
    results: dict[str, Any],
    benchmark: str,
) -> tuple[float, str, str] | None:
    preferred_keys = (benchmark, *results.keys())
    seen: set[str] = set()

    for result_key in preferred_keys:
        if result_key in seen:
            continue
        seen.add(str(result_key))
        metrics = results.get(result_key)
        if not isinstance(metrics, dict):
            continue
        for metric_name in metric_candidates(benchmark):
            score = numeric(metrics.get(metric_name))
            if score is not None:
                return score, metric_name, str(result_key)

    for result_key, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            metric = str(metric_name)
            if "stderr" in metric or metric == "alias":
                continue
            score = numeric(value)
            if score is not None:
                return score, metric, str(result_key)

    return None


def non_stderr_numeric_metrics(results: dict[str, Any], benchmark: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for result_key, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        key_prefix = benchmark if str(result_key) == benchmark else f"{benchmark}_{result_key}"
        for metric_name, value in metrics.items():
            metric = str(metric_name)
            if "stderr" in metric or metric == "alias":
                continue
            score = numeric(value)
            if score is None:
                continue
            column = f"{sanitize_column(key_prefix)}__{sanitize_column(metric)}"
            values[column] = score
    return values


def iter_result_records(input_dir: Path | str):
    input_dir = Path(input_dir)
    for result_path in input_dir.rglob("results_*.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        results = payload.get("results")
        if not isinstance(results, dict):
            continue

        context = model_and_benchmark(result_path, input_dir, payload)
        if context is None:
            continue
        model, benchmark = context
        timestamp = result_timestamp(result_path)
        selected = primary_score(results, benchmark)
        if selected is None:
            continue
        score, metric, result_key = selected
        yield {
            "model": model,
            "benchmark": benchmark,
            "score": score,
            "metric": metric,
            "result_key": result_key,
            "timestamp": timestamp,
            "result_path": result_path,
            "raw_metrics": non_stderr_numeric_metrics(results, benchmark),
        }


def ordered_benchmarks(found: set[str], preferred_order: tuple[str, ...]) -> list[str]:
    ordered = [name for name in preferred_order if name in found or preferred_order]
    extras = sorted(found.difference(ordered))
    return ordered + extras


def aggregate_lm_eval_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
    *,
    long_output_csv: Path | str | None = None,
    benchmark_order: tuple[str, ...] = DEFAULT_BENCHMARK_ORDER,
) -> Path:
    output_csv = Path(output_csv)
    long_output_csv = (
        Path(long_output_csv) if long_output_csv is not None else companion_long_output(output_csv)
    )
    latest: dict[tuple[str, str], dict[str, Any]] = {}

    for record in iter_result_records(input_dir):
        key = (str(record["model"]), str(record["benchmark"]))
        if key not in latest or float(record["timestamp"]) >= float(latest[key]["timestamp"]):
            latest[key] = record

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    found_benchmarks: set[str] = set()
    for (model, benchmark), record in latest.items():
        by_model.setdefault(model, {})[benchmark] = record
        found_benchmarks.add(benchmark)

    columns = ordered_benchmarks(found_benchmarks, benchmark_order)
    rows = []
    for model, records in by_model.items():
        scores = [float(records[name]["score"]) for name in columns if name in records]
        average = sum(scores) / len(scores) if scores else 0.0
        rows.append((average, model, records))
    rows.sort(key=lambda item: (item[0], item[1]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *columns, "average", "completed"])
        for average, model, records in rows:
            writer.writerow(
                [
                    model,
                    *(format_float(records[name]["score"]) if name in records else "" for name in columns),
                    format_float(average),
                    str(len(records)),
                ]
            )

    long_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with long_output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "benchmark", "score", "metric", "result_key", "result_path"])
        for _average, model, records in rows:
            for benchmark in columns:
                if benchmark not in records:
                    continue
                record = records[benchmark]
                writer.writerow(
                    [
                        model,
                        benchmark,
                        format_float(float(record["score"])),
                        record["metric"],
                        record["result_key"],
                        record["result_path"],
                    ]
                )

    return output_csv


def aggregate_lm_eval_raw_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_RAW_OUTPUT,
) -> Path:
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for record in iter_result_records(input_dir):
        model = str(record["model"])
        timestamp = float(record["timestamp"])
        for column, score in dict(record["raw_metrics"]).items():
            key = (model, column)
            if key not in latest or timestamp >= latest[key][0]:
                latest[key] = (timestamp, float(score))

    by_model: dict[str, dict[str, float]] = {}
    columns: set[str] = set()
    for (model, column), (_timestamp, score) in latest.items():
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


def aggregate_lm_eval_all_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
    *,
    long_output_csv: Path | str | None = None,
    raw_output_csv: Path | str | None = None,
) -> dict[str, Path]:
    output_csv = Path(output_csv)
    long_output_csv = (
        Path(long_output_csv) if long_output_csv is not None else companion_long_output(output_csv)
    )
    raw_output_csv = (
        Path(raw_output_csv) if raw_output_csv is not None else companion_raw_output(output_csv)
    )
    return {
        "summary": aggregate_lm_eval_results(
            input_dir,
            output_csv,
            long_output_csv=long_output_csv,
        ),
        "long_summary": long_output_csv,
        "raw_summary": aggregate_lm_eval_raw_results(input_dir, raw_output_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--long-output",
        type=Path,
        default=None,
        help=(
            "Path for the long model/benchmark/metric CSV. Defaults to a companion "
            "*_long_summary.csv next to --output."
        ),
    )
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
        "--no-raw-summary",
        action="store_true",
        help="Only write the benchmark-level summary and long summary CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = aggregate_lm_eval_results(args.input, args.output, long_output_csv=args.long_output)
    print(output)
    print(args.long_output or companion_long_output(output))
    if not args.no_raw_summary:
        raw_output = args.raw_output or companion_raw_output(output)
        raw_output = aggregate_lm_eval_raw_results(args.input, raw_output)
        print(raw_output)


if __name__ == "__main__":
    main()
