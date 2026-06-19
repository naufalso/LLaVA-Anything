from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_harmbench_score_report_writes_csv_summary(tmp_path: Path) -> None:
    report = load_script("scripts/utils/report_harmbench_scores.py")
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "reports" / "harmbench_scores.csv"
    results_path.write_text(
        json.dumps(
            {
                "behavior_a": [
                    {"label": 1, "generation": "unsafe", "test_case": "case 1"},
                    {"label": 0, "generation": "safe", "test_case": "case 2"},
                ],
                "behavior_b": [
                    {"label": "1", "generation": "unsafe", "test_case": "case 3"},
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = report.main(
        [
            "--result",
            f"demo={results_path}",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "run": "demo",
            "behaviors": "2",
            "samples": "3",
            "successes": "2",
            "failures": "1",
            "asr": "0.666667",
            "source": str(results_path),
        }
    ]


def test_harmbench_score_report_discovers_results_under_root(tmp_path: Path) -> None:
    report = load_script("scripts/utils/report_harmbench_scores.py")
    root = tmp_path / "harmbench"
    first = root / "model-a-text" / "llava-anything-text" / "DirectRequest" / "default" / "results" / "model_a.json"
    second = (
        root
        / "model-b-multimodal"
        / "llava-anything-multimodal"
        / "MultiModalDirectRequest"
        / "default"
        / "results"
        / "model_b.json"
    )
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps({"a": [{"label": 1}]}), encoding="utf-8")
    second.write_text(json.dumps({"b": [{"label": 0}, {"label": "yes"}]}), encoding="utf-8")

    output_path = tmp_path / "scores.csv"

    exit_code = report.main(["--results-root", str(root), "--output", str(output_path)])

    assert exit_code == 0
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["run"] for row in rows] == [
        "model-a-text/DirectRequest/default/model_a",
        "model-b-multimodal/MultiModalDirectRequest/default/model_b",
    ]
    assert [row["asr"] for row in rows] == ["1.000000", "0.500000"]
