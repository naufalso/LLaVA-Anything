from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoProcessor


CHECKPOINT = Path("checkpoints/llava-1.5-apretus-8b-clipl")
DATASETS = [
    ("vqav2_val", "lmms-lab/VQAv2", "validation", "answers"),
    ("ok_vqa_val2014", "lmms-lab/OK-VQA", "val2014", "answers"),
    ("textvqa_val", "lmms-lab/textvqa", "validation", "answers"),
    ("vizwiz_vqa_val", "lmms-lab/VizWiz-VQA", "val", "answers"),
]


def answers_from(doc: dict, key: str) -> list[str]:
    answers = doc.get(key)
    if answers is None:
        answers = doc.get("answer")
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]

    values: list[str] = []
    for item in answers:
        if isinstance(item, dict) and "answer" in item:
            values.append(str(item["answer"]))
        else:
            values.append(str(item))
    return values


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return sorted(values)[idx]


def main() -> None:
    selected_tasks = {
        task.strip()
        for task in os.environ.get("ANSWER_LENGTH_TASKS", "").split(",")
        if task.strip()
    }
    datasets = [item for item in DATASETS if not selected_tasks or item[0] in selected_tasks]
    if selected_tasks:
        missing_tasks = selected_tasks - {item[0] for item in DATASETS}
        if missing_tasks:
            raise ValueError(f"Unknown tasks in ANSWER_LENGTH_TASKS: {sorted(missing_tasks)}")

    print(
        "Inspecting tasks: "
        + ", ".join(task for task, *_ in datasets),
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    tokenizer = getattr(processor, "tokenizer", processor)

    for task, dataset_path, split, answer_key in datasets:
        print(f"\nTASK {task} ({dataset_path}:{split})", flush=True)
        dataset = load_dataset(dataset_path, split=split)

        all_answer_lengths: list[int] = []
        max_answer_lengths_by_doc: list[int] = []
        answers_over_16 = 0
        docs_over_16 = 0
        docs_over_32 = 0
        longest = {"answer": "", "tokens": -1, "doc_index": None}

        for doc_index, doc in enumerate(dataset):
            answers = answers_from(doc, answer_key)
            lengths = [
                len(tokenizer(answer, add_special_tokens=False).input_ids)
                for answer in answers
            ]
            if not lengths:
                continue

            all_answer_lengths.extend(lengths)
            doc_max = max(lengths)
            max_answer_lengths_by_doc.append(doc_max)
            answers_over_16 += sum(length > 16 for length in lengths)
            docs_over_16 += int(doc_max > 16)
            docs_over_32 += int(doc_max > 32)

            for answer, length in zip(answers, lengths, strict=True):
                if length > longest["tokens"]:
                    longest = {"answer": answer, "tokens": length, "doc_index": doc_index}

        total_answers = len(all_answer_lengths)
        total_docs = len(dataset)
        print(f"docs={total_docs} answer_strings={total_answers}")
        print(
            "answer_token_lengths "
            f"max={max(all_answer_lengths)} "
            f"p95={percentile(all_answer_lengths, 0.95)} "
            f"p99={percentile(all_answer_lengths, 0.99)}"
        )
        print(
            "per_doc_max_answer_token_lengths "
            f"max={max(max_answer_lengths_by_doc)} "
            f"p95={percentile(max_answer_lengths_by_doc, 0.95)} "
            f"p99={percentile(max_answer_lengths_by_doc, 0.99)}"
        )
        print(f"answers_gt_16={answers_over_16} ({answers_over_16 / total_answers:.4%})")
        print(f"docs_any_answer_gt_16={docs_over_16} ({docs_over_16 / total_docs:.4%})")
        print(f"docs_any_answer_gt_32={docs_over_32} ({docs_over_32 / total_docs:.4%})")
        print(
            "longest_answer "
            f"tokens={longest['tokens']} doc_index={longest['doc_index']} "
            f"answer={longest['answer']!r}"
        )


if __name__ == "__main__":
    main()
