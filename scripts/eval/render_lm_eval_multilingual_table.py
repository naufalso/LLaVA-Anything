#!/usr/bin/env python3
"""Render a paper-ready LaTeX table from lm-eval multilingual summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("output/reports/lm_eval_multilingual_summary.csv")
DEFAULT_OUTPUT = Path("output/reports/lm_eval_multilingual_table.tex")
REFERENCE_MODEL = "Apertus-8B-Instruct-2509"

TABLE_COLUMNS = (
    ("arc_multilingual", "ARC-M"),
    ("hellaswag_multilingual", "HellaSwag-M"),
    ("global_mmlu_gen_0shot", "Global-MMU"),
    ("agieval", "AGIEVal"),
    ("include_base_44_gen_0shot", "INCLUDE"),
    ("blend_sample", "BLEnD"),
    ("cultural_bench", "Cultural Bench"),
    ("multi_if", "Multi-IFEval"),
    ("truthfulqa_multilingual_mc2", "TruthfulQA-M"),
    ("average", "Avg."),
)

MODEL_DISPLAY_NAMES = {
    "Apertus-8B-Instruct-2509": "Apertus-8B-Instruct",
    "llava-1.5-apertus-8b-siglip-en": "LLaVA-Apertus-SigLIP-EN",
    "llava-1.5-apertus-8b-clipl-en": "LLaVA-Apertus-CLIP-L-EN",
    "llava-1.5-apertus-8b-siglip2-en": "LLaVA-Apertus-SigLIP2-EN",
    "llava-1.5-apertus-8b-clipl-palo": "LLaVA-Apertus-CLIP-L-PALO",
    "llava-1.5-apertus-8b-siglip-palo": "LLaVA-Apertus-SigLIP-PALO",
    "llava-1.5-apertus-8b-siglip2-palo": "LLaVA-Apertus-SigLIP2-PALO",
}

DEFAULT_CAPTION = (
    "Multilingual lm-evaluation-harness results. Scores are percentages. "
    "LLaVA rows report the score and the delta relative to Apertus in percentage points."
)
DEFAULT_LABEL = "tab:lm-eval-multilingual"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def score_percent(row: dict[str, str], column: str) -> float:
    return float(row[column]) * 100.0


def format_score(value: float) -> str:
    return f"{value:.2f}"


def format_delta(value: float, reference: float) -> str:
    return f"{value - reference:+.2f}"


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, latex_escape(model))


def sorted_rows(rows: list[dict[str, str]], reference_model: str) -> list[dict[str, str]]:
    reference = next((row for row in rows if row["model"] == reference_model), None)
    if reference is None:
        raise ValueError(f"Reference model not found in CSV: {reference_model}")

    others = [row for row in rows if row is not reference]
    others.sort(key=lambda row: float(row["average"]), reverse=True)
    return [reference, *others]


def render_score_cell(
    row: dict[str, str],
    reference: dict[str, str],
    column: str,
    *,
    is_reference: bool,
) -> str:
    value = score_percent(row, column)
    if is_reference:
        return format_score(value)

    reference_value = score_percent(reference, column)
    return rf"\score{{{format_score(value)}}}{{{format_delta(value, reference_value)}}}"


def render_table(
    rows: list[dict[str, str]],
    *,
    reference_model: str = REFERENCE_MODEL,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> str:
    ordered_rows = sorted_rows(rows, reference_model)
    reference = ordered_rows[0]
    headers = ["Model", *(label for _column, label in TABLE_COLUMNS)]
    column_spec = "l" + ("c" * len(TABLE_COLUMNS))
    lines = [
        "% Requires:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{graphicx}",
        "%",
        "% Move this command to the paper preamble if it is reused.",
        "\\newcommand{\\score}[2]{%",
        "\\begin{tabular}[t]{@{}l@{}}",
        "#1\\\\[-1pt]",
        "{\\scriptsize #2}",
        "\\end{tabular}}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]

    for row in ordered_rows:
        is_reference = row is reference
        cells = [display_model_name(row["model"])]
        cells.extend(
            render_score_cell(row, reference, column, is_reference=is_reference)
            for column, _label in TABLE_COLUMNS
        )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_lm_eval_multilingual_table(
    input_csv: Path | str = DEFAULT_INPUT,
    output_tex: Path | str = DEFAULT_OUTPUT,
    *,
    reference_model: str = REFERENCE_MODEL,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> Path:
    input_csv = Path(input_csv)
    output_tex = Path(output_tex)

    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    table = render_table(
        rows,
        reference_model=reference_model,
        caption=caption,
        label=label,
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(table)
    return output_tex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-model", default=REFERENCE_MODEL)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_lm_eval_multilingual_table(
        args.input,
        args.output,
        reference_model=args.reference_model,
        caption=args.caption,
        label=args.label,
    )
    print(output)


if __name__ == "__main__":
    main()
