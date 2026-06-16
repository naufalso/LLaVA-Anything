#!/usr/bin/env python3
"""Aggregate LMMS-Eval result JSON files into a paper-ready wide CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("output/lmms-eval")
DEFAULT_OUTPUT = Path("output/reports/lmms_eval_summary.csv")
DEFAULT_RAW_OUTPUT = Path("output/reports/lmms_eval_raw_summary.csv")
DEFAULT_UNNORMALIZED_OUTPUT = Path("output/reports/lmms_eval_unnormalized_summary.csv")
DEFAULT_UNNORMALIZED_RAW_OUTPUT = Path("output/reports/lmms_eval_unnormalized_raw_summary.csv")
MME_MAX_SCORE = 2800.0
MME_COGNITION_MAX_SCORE = 800.0
MME_PERCEPTION_MAX_SCORE = 2000.0


TASK_ALIASES = {
    "textvqa": "textvqa",
    "textvqa_val": "textvqa",
    "vizwiz": "vizwiz",
    "vizwiz_vqa_val": "vizwiz",
    "ok_vqa": "ok_vqa",
    "ok_vqa_val2014": "ok_vqa",
    "vqav2": "vqav2",
    "vqav2_val": "vqav2",
    "vqav2_20k": "vqav2",
    "pope": "pope",
    "mme": "mme",
    "seedbench": "seedbench",
    "mmbench": "mmbench",
    "mmbench_en_dev": "mmbench",
    "mmbench_en_dev_static": "mmbench",
    "mmmu": "mmmu_val",
    "mmmu_val": "mmmu_val",
    "mmstar": "mmstar",
}


BENCHMARK_PRIMARY_METRICS = {
    "mmmu_val": ("mmmu_acc,none",),
    "mmstar": ("average,none",),
}


PRIMARY_METRICS = (
    "exact_match,none",
    "pope_accuracy,none",
    "seed_all,none",
    "gpt_eval_score,none",
    "accuracy,none",
    "acc,none",
)


def clean_model_name(value: str | None, path: Path, input_dir: Path) -> str:
    if value:
        name = Path(value).name
        if name:
            return name

    try:
        rel_parts = path.relative_to(input_dir).parts
    except ValueError:
        rel_parts = path.parts

    for part in rel_parts:
        if part not in TASK_ALIASES and part not in {"submissions"}:
            if not part.endswith("_results.json"):
                return part
    return path.stem.replace("_results", "")


def benchmark_name(task_name: str) -> str:
    return TASK_ALIASES.get(task_name, task_name)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_column(value: str) -> str:
    value = value.split(",", 1)[0]
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_primary_metric(metric: str, score: float, raw: bool) -> float:
    if metric == "gpt_eval_score,none" and not raw:
        return score / 100.0
    return score


def primary_score(metrics: dict[str, Any], benchmark: str, raw: bool = False) -> float | None:
    if benchmark == "mme":
        cognition = numeric(metrics.get("mme_cognition_score,none"))
        perception = numeric(metrics.get("mme_perception_score,none"))
        if cognition is not None and perception is not None:
            score = cognition + perception
            return score if raw else score / MME_MAX_SCORE

    for metric in BENCHMARK_PRIMARY_METRICS.get(benchmark, ()):
        score = numeric(metrics.get(metric))
        if score is not None:
            return normalize_primary_metric(metric, score, raw)

    for metric in PRIMARY_METRICS:
        score = numeric(metrics.get(metric))
        if score is not None:
            return normalize_primary_metric(metric, score, raw)

    for metric, value in metrics.items():
        if "stderr" in metric or metric.endswith("_stderr,none"):
            continue
        score = numeric(value)
        if score is not None:
            return score
    return None


def non_stderr_numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    values = {}
    for metric, value in metrics.items():
        metric_name = sanitize_column(str(metric))
        if not metric_name or "stderr" in metric_name or metric_name == "alias":
            continue
        score = numeric(value)
        if score is not None:
            values[metric_name] = score
    return values


def raw_metric_column(benchmark: str, metric_name: str) -> str:
    if metric_name.startswith(f"{benchmark}_"):
        return metric_name
    return f"{benchmark}_{metric_name}"


def normalize_raw_metric(column: str, score: float) -> float:
    if column == "mmbench_gpt_eval_score":
        return score / 100.0
    if column == "mme_cognition_score":
        return score / MME_COGNITION_MAX_SCORE
    if column == "mme_perception_score":
        return score / MME_PERCEPTION_MAX_SCORE
    return score


def effective_samples(payload: dict[str, Any], task_name: str) -> float | None:
    samples = payload.get("n-samples")
    if not isinstance(samples, dict):
        return None

    task_samples = samples.get(task_name)
    if isinstance(task_samples, dict):
        return numeric(task_samples.get("effective"))
    return numeric(task_samples)


def valid_task_result(
    payload: dict[str, Any],
    task_name: str,
    min_effective_samples: int,
) -> bool:
    sample_count = effective_samples(payload, task_name)
    return sample_count is None or sample_count >= min_effective_samples


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def companion_raw_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_raw_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_raw_summary{output_csv.suffix}")


def companion_unnormalized_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(f"{output_csv.stem[:-8]}_unnormalized_summary{output_csv.suffix}")
    return output_csv.with_name(f"{output_csv.stem}_unnormalized_summary{output_csv.suffix}")


def companion_unnormalized_raw_output(output_csv: Path) -> Path:
    if output_csv.stem.endswith("_summary"):
        return output_csv.with_name(
            f"{output_csv.stem[:-8]}_unnormalized_raw_summary{output_csv.suffix}"
        )
    return output_csv.with_name(f"{output_csv.stem}_unnormalized_raw_summary{output_csv.suffix}")


def aggregate_lmms_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
    *,
    raw: bool = False,
    min_effective_samples: int = 2,
) -> Path:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for result_path in input_dir.rglob("*_results.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        results = payload.get("results")
        if not isinstance(results, dict):
            continue

        model = clean_model_name(
            payload.get("model_name") or payload.get("model_name_sanitized"),
            result_path,
            input_dir,
        )
        mtime = result_path.stat().st_mtime

        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            if not valid_task_result(payload, str(task_name), min_effective_samples):
                continue
            benchmark = benchmark_name(str(task_name))
            score = primary_score(metrics, benchmark, raw=raw)
            if score is None:
                continue
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


def sibling_model_for_submission(
    submission_path: Path,
    input_dir: Path,
    min_effective_samples: int,
) -> tuple[str, str] | None:
    if submission_path.parent.name != "submissions":
        return None

    benchmark_dir = submission_path.parent.parent
    benchmark = benchmark_name(benchmark_dir.name)
    candidates = []

    for result_path in benchmark_dir.rglob("*_results.json"):
        if "submissions" in result_path.parts:
            continue
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        results = payload.get("results")
        if not isinstance(results, dict):
            continue

        for task_name in results:
            if benchmark_name(str(task_name)) != benchmark:
                continue
            if not valid_task_result(payload, str(task_name), min_effective_samples):
                continue
            model = clean_model_name(
                payload.get("model_name") or payload.get("model_name_sanitized"),
                result_path,
                input_dir,
            )
            candidates.append((result_path.stat().st_mtime, model))

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1][1], benchmark


def aggregate_lmms_raw_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_RAW_OUTPUT,
    *,
    normalize: bool = True,
    min_effective_samples: int = 2,
) -> Path:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    latest: dict[tuple[str, str], tuple[float, float]] = {}

    for result_path in input_dir.rglob("*_results.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        results = payload.get("results")
        mtime = result_path.stat().st_mtime

        if isinstance(results, dict):
            model = clean_model_name(
                payload.get("model_name") or payload.get("model_name_sanitized"),
                result_path,
                input_dir,
            )
            for task_name, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                task_name = str(task_name)
                if not valid_task_result(payload, task_name, min_effective_samples):
                    continue
                benchmark = benchmark_name(task_name)
                for metric_name, score in non_stderr_numeric_metrics(metrics).items():
                    column = raw_metric_column(benchmark, metric_name)
                    if normalize:
                        score = normalize_raw_metric(column, score)
                    key = (model, column)
                    if key not in latest or mtime >= latest[key][0]:
                        latest[key] = (mtime, score)
            continue

        if not isinstance(payload, dict) or "overall_acc" not in payload:
            continue

        context = sibling_model_for_submission(result_path, input_dir, min_effective_samples)
        if context is None:
            continue
        model, benchmark = context

        overall = numeric(payload.get("overall_acc"))
        if overall is not None:
            latest[(model, f"{benchmark}_overall_acc")] = (mtime, overall)

        for group_name in ("category_acc", "l2_category_acc"):
            group = payload.get(group_name)
            if not isinstance(group, dict):
                continue
            group_column = sanitize_column(group_name.replace("_acc", ""))
            for category, value in group.items():
                score = numeric(value)
                if score is None:
                    continue
                column = f"{benchmark}_{group_column}_{sanitize_column(str(category))}"
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


def aggregate_lmms_all_results(
    input_dir: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
    *,
    raw_output_csv: Path | str | None = None,
    unnormalized_output_csv: Path | str | None = None,
    unnormalized_raw_output_csv: Path | str | None = None,
    min_effective_samples: int = 2,
) -> dict[str, Path]:
    output_csv = Path(output_csv)
    raw_output_csv = Path(raw_output_csv) if raw_output_csv is not None else companion_raw_output(output_csv)
    unnormalized_output_csv = (
        Path(unnormalized_output_csv)
        if unnormalized_output_csv is not None
        else companion_unnormalized_output(output_csv)
    )
    unnormalized_raw_output_csv = (
        Path(unnormalized_raw_output_csv)
        if unnormalized_raw_output_csv is not None
        else companion_unnormalized_raw_output(output_csv)
    )

    outputs = {
        "summary": aggregate_lmms_results(
            input_dir,
            output_csv,
            raw=False,
            min_effective_samples=min_effective_samples,
        ),
        "raw_summary": aggregate_lmms_raw_results(
            input_dir,
            raw_output_csv,
            normalize=True,
            min_effective_samples=min_effective_samples,
        ),
        "unnormalized_summary": aggregate_lmms_results(
            input_dir,
            unnormalized_output_csv,
            raw=True,
            min_effective_samples=min_effective_samples,
        ),
        "unnormalized_raw_summary": aggregate_lmms_raw_results(
            input_dir,
            unnormalized_raw_output_csv,
            normalize=False,
            min_effective_samples=min_effective_samples,
        ),
    }
    return outputs


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
        "--unnormalized-output",
        type=Path,
        default=None,
        help=(
            "Path for the unnormalized benchmark-level summary CSV. Defaults to a "
            "*_unnormalized_summary.csv companion next to --output."
        ),
    )
    parser.add_argument(
        "--unnormalized-raw-output",
        type=Path,
        default=None,
        help=(
            "Path for the unnormalized expanded raw metrics CSV. Defaults to a "
            "*_unnormalized_raw_summary.csv companion next to --output."
        ),
    )
    parser.add_argument(
        "--no-raw-summary",
        action="store_true",
        help="Skip both normalized and unnormalized expanded raw metrics CSVs.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Legacy mode: write --output as an unnormalized benchmark summary. "
            "Without this flag, the script writes normalized and unnormalized reports."
        ),
    )
    parser.add_argument(
        "--min-effective-samples",
        type=int,
        default=2,
        help=(
            "Skip LMMS task results with an explicit effective sample count below this "
            "threshold. Files without sample counts are kept. Default: 2."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw:
        output = aggregate_lmms_results(
            args.input,
            args.output,
            raw=True,
            min_effective_samples=args.min_effective_samples,
        )
        print(output)
        if not args.no_raw_summary:
            raw_output = args.raw_output or companion_raw_output(output)
            raw_output = aggregate_lmms_raw_results(
                args.input,
                raw_output,
                normalize=False,
                min_effective_samples=args.min_effective_samples,
            )
            print(raw_output)
        return

    output = aggregate_lmms_results(
        args.input,
        args.output,
        raw=False,
        min_effective_samples=args.min_effective_samples,
    )
    print(output)

    if not args.no_raw_summary:
        raw_output = args.raw_output or companion_raw_output(output)
        raw_output = aggregate_lmms_raw_results(
            args.input,
            raw_output,
            normalize=True,
            min_effective_samples=args.min_effective_samples,
        )
        print(raw_output)

    unnormalized_output = args.unnormalized_output or companion_unnormalized_output(output)
    unnormalized_output = aggregate_lmms_results(
        args.input,
        unnormalized_output,
        raw=True,
        min_effective_samples=args.min_effective_samples,
    )
    print(unnormalized_output)

    if not args.no_raw_summary:
        unnormalized_raw_output = (
            args.unnormalized_raw_output or companion_unnormalized_raw_output(output)
        )
        unnormalized_raw_output = aggregate_lmms_raw_results(
            args.input,
            unnormalized_raw_output,
            normalize=False,
            min_effective_samples=args.min_effective_samples,
        )
        print(unnormalized_raw_output)


if __name__ == "__main__":
    main()
