#!/usr/bin/env python3
"""Summarize HarmBench classifier results into a reusable score report."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path("output/reports/harmbench/harmbench_scores.csv")
DEFAULT_RESULTS_ROOT = Path("output/harmbench")


@dataclass(frozen=True)
class HarmBenchSummary:
    name: str
    path: Path
    behaviors: int
    samples: int
    successes: int

    @property
    def failures(self) -> int:
        return self.samples - self.successes

    @property
    def asr(self) -> float:
        return self.successes / self.samples if self.samples else 0.0


def is_success_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "yes", "true"}
    return False


def iter_result_items(results: dict[str, Any], path: Path) -> Iterable[dict[str, Any]]:
    for behavior_id, values in results.items():
        if not isinstance(values, list):
            raise ValueError(f"Expected list of result records for {behavior_id!r} in {path}")
        for item in values:
            if not isinstance(item, dict) or "label" not in item:
                raise ValueError(f"Expected result record with a label for {behavior_id!r} in {path}")
            yield item


def summarize_result(name: str, path: Path) -> HarmBenchSummary:
    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, dict):
        raise ValueError(f"Expected JSON object in {path}")

    labels = [item["label"] for item in iter_result_items(results, path)]
    successes = sum(is_success_label(label) for label in labels)
    return HarmBenchSummary(
        name=name,
        path=path,
        behaviors=len(results),
        samples=len(labels),
        successes=successes,
    )


def parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Result name cannot be empty")
    return name, Path(path)


def infer_result_name(results_root: Path, path: Path) -> str:
    relative = path.relative_to(results_root)
    parts = relative.parts
    if len(parts) >= 6 and parts[-2] == "results":
        return "/".join((parts[0], parts[2], parts[3], path.stem))
    return relative.with_suffix("").as_posix()


def discover_results(results_root: Path) -> list[tuple[str, Path]]:
    return [
        (infer_result_name(results_root, path), path)
        for path in sorted(results_root.glob("**/results/*.json"))
    ]


def write_report(summaries: list[HarmBenchSummary], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "behaviors", "samples", "successes", "failures", "asr", "source"],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "run": summary.name,
                    "behaviors": summary.behaviors,
                    "samples": summary.samples,
                    "successes": summary.successes,
                    "failures": summary.failures,
                    "asr": f"{summary.asr:.6f}",
                    "source": summary.path,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_result_arg,
        help="Result file to summarize as NAME=PATH. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV report path. Defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root to scan for HarmBench result files when --result is not provided. Defaults to {DEFAULT_RESULTS_ROOT}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = args.result if args.result else discover_results(args.results_root)
    summaries = [summarize_result(name, path) for name, path in results]
    write_report(summaries, args.output)
    print(f"Wrote HarmBench score report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
