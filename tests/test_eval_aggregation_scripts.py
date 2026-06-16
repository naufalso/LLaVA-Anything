import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_lmms_aggregation_normalizes_and_sorts_by_average(tmp_path):
    lmms = load_script("scripts/eval/aggregate_lmms_results.py")
    input_dir = tmp_path / "lmms"
    output_csv = tmp_path / "lmms_summary.csv"

    (input_dir / "model_a" / "textvqa").mkdir(parents=True)
    (input_dir / "model_a" / "mmbench").mkdir(parents=True)
    (input_dir / "model_b" / "textvqa").mkdir(parents=True)
    (input_dir / "model_b" / "mme").mkdir(parents=True)

    (input_dir / "model_a" / "textvqa" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {"textvqa_val": {"exact_match,none": 0.2}},
            }
        )
    )
    (input_dir / "model_a" / "mmbench" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {"mmbench_en_dev": {"gpt_eval_score,none": 50.0}},
            }
        )
    )
    (input_dir / "model_b" / "textvqa" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_b",
                "results": {"textvqa_val": {"exact_match,none": 0.7}},
            }
        )
    )
    (input_dir / "model_b" / "mme" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_b",
                "results": {
                    "mme": {
                        "mme_cognition_score,none": 700.0,
                        "mme_perception_score,none": 700.0,
                    }
                },
            }
        )
    )

    lmms.aggregate_lmms_results(input_dir, output_csv)

    rows = read_csv(output_csv)
    assert [row["model"] for row in rows] == ["model_a", "model_b"]
    assert rows[0]["textvqa"] == "0.2"
    assert rows[0]["mmbench"] == "0.5"
    assert rows[0]["average"] == "0.35"
    assert rows[1]["mme"] == "0.5"
    assert rows[1]["average"] == "0.6"


def test_lmms_aggregation_prefers_benchmark_average_metrics(tmp_path):
    lmms = load_script("scripts/eval/aggregate_lmms_results.py")
    input_dir = tmp_path / "lmms"
    output_csv = tmp_path / "lmms_summary.csv"

    (input_dir / "model_a" / "mmstar").mkdir(parents=True)
    (input_dir / "model_a" / "mmmu").mkdir(parents=True)

    (input_dir / "model_a" / "mmstar" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {
                    "mmstar": {
                        "coarse perception,none": 0.9,
                        "average,none": 0.4,
                    }
                },
            }
        )
    )
    (input_dir / "model_a" / "mmmu" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {
                    "mmmu_val": {
                        "science,none": 0.8,
                        "mmmu_acc,none": 0.3,
                    }
                },
            }
        )
    )

    lmms.aggregate_lmms_results(input_dir, output_csv)

    row = read_csv(output_csv)[0]
    assert row["mmstar"] == "0.4"
    assert row["mmmu_val"] == "0.3"
    assert row["average"] == "0.35"


def test_lmms_raw_summary_expands_available_metrics_and_categories(tmp_path):
    lmms = load_script("scripts/eval/aggregate_lmms_results.py")
    input_dir = tmp_path / "lmms"
    output_csv = tmp_path / "lmms_raw_summary.csv"

    (input_dir / "suite" / "model_a" / "pope" / "models__model_a").mkdir(parents=True)
    (input_dir / "suite" / "model_a" / "mmbench" / "models__model_a").mkdir(parents=True)
    (input_dir / "suite" / "model_a" / "mmbench" / "submissions").mkdir(parents=True)

    (input_dir / "suite" / "model_a" / "pope" / "models__model_a" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "n-samples": {"pope": {"effective": 100}},
                "results": {
                    "pope": {
                        "pope_accuracy,none": 0.8,
                        "pope_precision,none": 0.7,
                        "pope_accuracy_stderr,none": 0.01,
                    }
                },
            }
        )
    )
    (input_dir / "suite" / "model_a" / "mmbench" / "models__model_a" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "n-samples": {"mmbench_en_dev": {"effective": 100}},
                "results": {"mmbench_en_dev": {"gpt_eval_score,none": 50.0}},
            }
        )
    )
    (input_dir / "suite" / "model_a" / "mmbench" / "submissions" / "mmbench_en_dev_results.json").write_text(
        json.dumps(
            {
                "overall_acc": 0.5,
                "category_acc": {"action_recognition": 0.4},
                "l2_category_acc": {"attribute_reasoning": 0.6},
            }
        )
    )

    lmms.aggregate_lmms_raw_results(input_dir, output_csv, normalize=False)

    rows = read_csv(output_csv)
    assert len(rows) == 1
    assert rows[0]["model"] == "model_a"
    assert rows[0]["pope_accuracy"] == "0.8"
    assert rows[0]["pope_precision"] == "0.7"
    assert "pope_accuracy_stderr" not in rows[0]
    assert rows[0]["mmbench_gpt_eval_score"] == "50"
    assert rows[0]["mmbench_overall_acc"] == "0.5"
    assert rows[0]["mmbench_category_action_recognition"] == "0.4"
    assert rows[0]["mmbench_l2_category_attribute_reasoning"] == "0.6"


def test_lmms_raw_summary_can_normalize_mixed_scale_metrics(tmp_path):
    lmms = load_script("scripts/eval/aggregate_lmms_results.py")
    input_dir = tmp_path / "lmms"
    normalized_csv = tmp_path / "lmms_raw_summary.csv"
    unnormalized_csv = tmp_path / "lmms_unnormalized_raw_summary.csv"

    (input_dir / "model_a" / "mmbench").mkdir(parents=True)
    (input_dir / "model_a" / "mme").mkdir(parents=True)

    (input_dir / "model_a" / "mmbench" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {"mmbench_en_dev": {"gpt_eval_score,none": 50.0}},
            }
        )
    )
    (input_dir / "model_a" / "mme" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {
                    "mme": {
                        "mme_cognition_score,none": 400.0,
                        "mme_perception_score,none": 1000.0,
                    }
                },
            }
        )
    )

    lmms.aggregate_lmms_raw_results(input_dir, normalized_csv)
    lmms.aggregate_lmms_raw_results(input_dir, unnormalized_csv, normalize=False)

    normalized = read_csv(normalized_csv)[0]
    unnormalized = read_csv(unnormalized_csv)[0]
    assert normalized["mmbench_gpt_eval_score"] == "0.5"
    assert normalized["mme_cognition_score"] == "0.5"
    assert normalized["mme_perception_score"] == "0.5"
    assert unnormalized["mmbench_gpt_eval_score"] == "50"
    assert unnormalized["mme_cognition_score"] == "400"
    assert unnormalized["mme_perception_score"] == "1000"


def test_lmms_all_outputs_writes_normalized_and_unnormalized_pairs(tmp_path):
    lmms = load_script("scripts/eval/aggregate_lmms_results.py")
    input_dir = tmp_path / "lmms"
    summary_csv = tmp_path / "custom_summary.csv"

    (input_dir / "model_a" / "mmbench").mkdir(parents=True)
    (input_dir / "model_a" / "mme").mkdir(parents=True)

    (input_dir / "model_a" / "mmbench" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {"mmbench_en_dev": {"gpt_eval_score,none": 50.0}},
            }
        )
    )
    (input_dir / "model_a" / "mme" / "run_results.json").write_text(
        json.dumps(
            {
                "model_name": "/models/model_a",
                "results": {
                    "mme": {
                        "mme_cognition_score,none": 400.0,
                        "mme_perception_score,none": 1000.0,
                    }
                },
            }
        )
    )

    outputs = lmms.aggregate_lmms_all_results(input_dir, summary_csv)

    assert outputs == {
        "summary": summary_csv,
        "raw_summary": tmp_path / "custom_raw_summary.csv",
        "unnormalized_summary": tmp_path / "custom_unnormalized_summary.csv",
        "unnormalized_raw_summary": tmp_path / "custom_unnormalized_raw_summary.csv",
    }
    for output in outputs.values():
        assert output.exists()

    summary = read_csv(outputs["summary"])[0]
    raw_summary = read_csv(outputs["raw_summary"])[0]
    unnormalized_summary = read_csv(outputs["unnormalized_summary"])[0]
    unnormalized_raw_summary = read_csv(outputs["unnormalized_raw_summary"])[0]

    assert summary["mmbench"] == "0.5"
    assert summary["mme"] == "0.5"
    assert unnormalized_summary["mmbench"] == "50"
    assert unnormalized_summary["mme"] == "1400"
    assert raw_summary["mmbench_gpt_eval_score"] == "0.5"
    assert raw_summary["mme_cognition_score"] == "0.5"
    assert unnormalized_raw_summary["mmbench_gpt_eval_score"] == "50"
    assert unnormalized_raw_summary["mme_cognition_score"] == "400"


def test_vlmevalkit_aggregation_uses_aggregate_files_only(tmp_path):
    vlm = load_script("scripts/eval/aggregate_vlmevalkit_results.py")
    input_dir = tmp_path / "vlmevalkit"
    output_csv = tmp_path / "vlmevalkit_summary.csv"
    input_dir.mkdir()

    (input_dir / "model_a_MMMB_acc.csv").write_text(
        "DATASET,Overall\nMMMB_ar,0.2\nMMMB_cn,0.4\n"
    )
    (input_dir / "model_a_MMMB_ar_acc.csv").write_text(
        "split,Overall\nignored,1.0\n"
    )
    (input_dir / "model_b_MTL_MMBench_DEV_acc.csv").write_text(
        "DATASET,Overall\nMMBench_dev_ar,0.8\nMMBench_dev_cn,0.6\n"
    )

    vlm.aggregate_vlmevalkit_results(input_dir, output_csv)

    rows = read_csv(output_csv)
    assert [row["model"] for row in rows] == ["model_a", "model_b"]
    assert rows[0]["MMMB"] == "0.3"
    assert rows[0]["MTL_MMBench_DEV"] == ""
    assert rows[0]["average"] == "0.3"
    assert rows[1]["MMMB"] == ""
    assert rows[1]["MTL_MMBench_DEV"] == "0.7"
    assert rows[1]["average"] == "0.7"


def test_vlmevalkit_raw_summary_expands_languages_and_categories(tmp_path):
    vlm = load_script("scripts/eval/aggregate_vlmevalkit_results.py")
    input_dir = tmp_path / "vlmevalkit"
    output_csv = tmp_path / "vlmevalkit_raw_summary.csv"
    input_dir.mkdir()

    (input_dir / "model_a_MMMB_acc.csv").write_text(
        "DATASET,Overall,Scene Understanding,OCR\n"
        "MMMB_en,0.7,0.8,0.9\n"
        "MMMB_cn,0.5,0.6,0.4\n"
    )
    (input_dir / "model_a_MTL_MMBench_DEV_acc.csv").write_text(
        "DATASET,Overall,action_recognition\n"
        "MMBench_dev_en,0.6,0.3\n"
    )

    vlm.aggregate_vlmevalkit_raw_results(input_dir, output_csv)

    rows = read_csv(output_csv)
    assert len(rows) == 1
    assert rows[0]["model"] == "model_a"
    assert rows[0]["mmmb_en"] == "0.7"
    assert rows[0]["mmmb_en_scene_understanding"] == "0.8"
    assert rows[0]["mmmb_en_ocr"] == "0.9"
    assert rows[0]["mmmb_cn"] == "0.5"
    assert rows[0]["mtl_mmbench_dev_en"] == "0.6"
    assert rows[0]["mtl_mmbench_dev_en_action_recognition"] == "0.3"


def test_vlmevalkit_mm_summary_keeps_language_overalls_only(tmp_path):
    vlm = load_script("scripts/eval/aggregate_vlmevalkit_results.py")
    input_dir = tmp_path / "vlmevalkit"
    output_csv = tmp_path / "vlmevalkit_mm_summary.csv"
    input_dir.mkdir()

    (input_dir / "model_a_MMMB_acc.csv").write_text(
        "DATASET,Overall,Scene Understanding\n"
        "MMMB_en,0.2,0.9\n"
        "MMMB_cn,0.4,0.8\n"
    )
    (input_dir / "model_b_MTL_MMBench_DEV_acc.csv").write_text(
        "DATASET,Overall,action_recognition\n"
        "MMBench_dev_en,0.8,0.3\n"
        "MMBench_dev_cn,0.6,0.4\n"
    )

    vlm.aggregate_vlmevalkit_mm_results(input_dir, output_csv)

    rows = read_csv(output_csv)
    assert [row["model"] for row in rows] == ["model_a", "model_b"]
    assert rows[0]["mmmb_en"] == "0.2"
    assert rows[0]["mmmb_cn"] == "0.4"
    assert rows[0]["mtl_mmbench_dev_en"] == ""
    assert rows[0]["average"] == "0.3"
    assert rows[1]["mmmb_en"] == ""
    assert rows[1]["mtl_mmbench_dev_en"] == "0.8"
    assert rows[1]["mtl_mmbench_dev_cn"] == "0.6"
    assert rows[1]["average"] == "0.7"
    assert "mmmb_en_scene_understanding" not in rows[0]


def test_lm_eval_multilingual_summary_selects_primary_metrics_and_latest_result(tmp_path):
    lm_eval = load_script("scripts/eval/aggregate_lm_eval_results.py")
    input_dir = tmp_path / "lm-eval" / "multilingual-lm-eval"
    output_csv = tmp_path / "lm_eval_multilingual_summary.csv"
    long_csv = tmp_path / "lm_eval_multilingual_long_summary.csv"

    old_dir = input_dir / "model_a" / "arc_multilingual" / "run_old"
    new_dir = input_dir / "model_a" / "arc_multilingual" / "run_new"
    multi_if_dir = input_dir / "model_a" / "multi_if" / "run"
    truthfulqa_dir = input_dir / "model_b" / "truthfulqa_multilingual_mc2" / "run"
    for path in (old_dir, new_dir, multi_if_dir, truthfulqa_dir):
        path.mkdir(parents=True)

    old_result = old_dir / "results_2026-01-01T00-00-00.json"
    old_result.write_text(
        json.dumps(
            {
                "results": {
                    "arc_multilingual": {
                        "acc,none": 0.1,
                        "acc_norm,none": 0.2,
                    }
                }
            }
        )
    )
    new_result = new_dir / "results_2026-01-02T00-00-00.json"
    new_result.write_text(
        json.dumps(
            {
                "results": {
                    "arc_multilingual": {
                        "acc,none": 0.3,
                        "acc_norm,none": 0.4,
                    }
                }
            }
        )
    )
    multi_if_result = multi_if_dir / "results_2026-01-03T00-00-00.json"
    multi_if_result.write_text(
        json.dumps(
            {
                "results": {
                    "multi_if": {
                        "prompt_level_strict_acc,none": 0.5,
                        "inst_level_strict_acc,none": 0.7,
                    }
                }
            }
        )
    )
    truthfulqa_result = truthfulqa_dir / "results_2026-01-04T00-00-00.json"
    truthfulqa_result.write_text(
        json.dumps(
            {
                "results": {
                    "truthfulqa_multilingual_mc2": {
                        "acc,none": 0.6,
                        "acc_stderr,none": 0.01,
                    }
                }
            }
        )
    )

    lm_eval.aggregate_lm_eval_results(input_dir, output_csv, long_output_csv=long_csv)

    rows = read_csv(output_csv)
    assert [row["model"] for row in rows] == ["model_a", "model_b"]
    assert rows[0]["arc_multilingual"] == "0.4"
    assert rows[0]["multi_if"] == "0.5"
    assert rows[0]["truthfulqa_multilingual_mc2"] == ""
    assert rows[0]["average"] == "0.45"
    assert rows[0]["completed"] == "2"
    assert rows[1]["truthfulqa_multilingual_mc2"] == "0.6"
    assert rows[1]["average"] == "0.6"

    long_rows = read_csv(long_csv)
    arc_rows = [
        row
        for row in long_rows
        if row["model"] == "model_a" and row["benchmark"] == "arc_multilingual"
    ]
    assert len(arc_rows) == 1
    assert arc_rows[0]["score"] == "0.4"
    assert arc_rows[0]["metric"] == "acc_norm,none"
    assert arc_rows[0]["result_path"].endswith("results_2026-01-02T00-00-00.json")


def test_lm_eval_multilingual_latex_table_uses_apertus_reference_and_sorts_llava(tmp_path):
    table = load_script("scripts/eval/render_lm_eval_multilingual_table.py")
    input_csv = tmp_path / "lm_eval_multilingual_summary.csv"
    output_tex = tmp_path / "lm_eval_multilingual_table.tex"

    input_csv.write_text(
        "\n".join(
            [
                "model,agieval,arc_multilingual,blend_sample,cultural_bench,global_mmlu_gen_0shot,hellaswag_multilingual,include_base_44_gen_0shot,multi_if,truthfulqa_multilingual_mc2,average,completed",
                "llava-1.5-apertus-8b-low,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.35,9",
                "Apertus-8B-Instruct-2509,0.40,0.50,0.60,0.70,0.80,0.90,0.30,0.20,0.10,0.50,9",
                "llava-1.5-apertus-8b-high,0.41,0.52,0.63,0.74,0.85,0.96,0.37,0.28,0.19,0.55,9",
            ]
        )
        + "\n"
    )

    table.render_lm_eval_multilingual_table(input_csv, output_tex)

    rendered = output_tex.read_text()
    assert "Model & ARC-M & HellaSwag-M & Global-MMU & AGIEVal" in rendered
    assert rendered.index("Apertus-8B-Instruct &") < rendered.index(
        "llava-1.5-apertus-8b-high"
    )
    assert rendered.index("llava-1.5-apertus-8b-high") < rendered.index(
        "llava-1.5-apertus-8b-low"
    )
    assert (
        "llava-1.5-apertus-8b-high & \\score{52.00}{+2.00} & "
        "\\score{96.00}{+6.00}"
    ) in rendered
    assert "llava-1.5-apertus-8b-low & \\score{32.00}{-18.00}" in rendered
    assert "\\newcommand{\\score}[2]" in rendered
