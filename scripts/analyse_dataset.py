"""Analyze LLaVA-Anything training datasets from the same YAML used for training.

The analyzer loads the model processor/tokenizer, renders records with the same
chat template used by training, expands image tokens with the processor's image
geometry, and reports distributions that are useful when choosing
``training.model_max_length``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is present in normal training envs.

    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else []


_RUNTIME_IMPORTS_LOADED = False
_WORKER_PROCESSOR: Any | None = None
_WORKER_OPTIONS: dict[str, Any] = {}


SUMMARY_PERCENTILES = [50, 75, 90, 95, 99, 99.5, 100]
DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 6144, 8192, 12288, 16384, 32768]
DEFAULT_IMAGE_OPEN_TIMEOUT_SECONDS = 30.0


class ImageOpenTimeoutError(TimeoutError):
    """Raised when PIL takes too long to read image metadata."""


def _load_runtime_imports() -> None:
    """Import training/runtime dependencies only when analysis actually runs."""

    global _RUNTIME_IMPORTS_LOADED
    global LlavaAnythingProcessor
    global config_from_yaml_dict, load_yaml, processor_from_yaml_dict
    global _load_json_records, _preview_record, _resolve_model_max_length

    if _RUNTIME_IMPORTS_LOADED:
        return

    try:
        import llava_anything  # noqa: F401 - registers auto classes
        from llava_anything.builder import (
            config_from_yaml_dict as imported_config_from_yaml_dict,
        )
        from llava_anything.builder import load_yaml as imported_load_yaml
        from llava_anything.builder import (
            processor_from_yaml_dict as imported_processor_from_yaml_dict,
        )
        from llava_anything.processing_llava_anything import (
            LlavaAnythingProcessor as ImportedLlavaAnythingProcessor,
        )
        from llava_anything.dataset import (
            _load_json_records as imported_load_json_records,
        )
        from llava_anything.dataset import _preview_record as imported_preview_record
        from llava_anything.dataset import (
            _resolve_model_max_length as imported_resolve_model_max_length,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Dataset analysis requires the same runtime dependencies as training. "
            "Install the project environment with torch, transformers, pillow, and pyyaml."
        ) from exc

    LlavaAnythingProcessor = ImportedLlavaAnythingProcessor
    config_from_yaml_dict = imported_config_from_yaml_dict
    load_yaml = imported_load_yaml
    processor_from_yaml_dict = imported_processor_from_yaml_dict
    _load_json_records = imported_load_json_records
    _preview_record = imported_preview_record
    _resolve_model_max_length = imported_resolve_model_max_length
    _RUNTIME_IMPORTS_LOADED = True


RECORD_FIELDS = [
    "record_index",
    "record_id",
    "status",
    "skip_reason",
    "usable_without_context_cap",
    "usable_for_configured_training",
    "has_training_image",
    "declared_image_count",
    "rendered_image_markers",
    "conversation_turns",
    "image_path",
    "image_width",
    "image_height",
    "image_pixels",
    "image_aspect_ratio",
    "image_tokens",
    "full_tokens",
    "text_tokens",
    "prompt_tokens",
    "prompt_text_tokens",
    "target_tokens",
    "target_text_tokens",
    "image_tokens_in_prompt",
    "image_tokens_in_target",
    "fits_model_max_length",
    "needs_truncation_at_model_max_length",
    "length_over_model_max_length",
    "truncated_total_tokens",
]


BROKEN_IMAGE_FIELDS = [
    "record_index",
    "record_id",
    "skip_reason",
    "image_path",
    "has_training_image",
    "declared_image_count",
    "rendered_image_markers",
    "conversation_turns",
    "image_width",
    "image_height",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _optional_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _optional_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _round_up(value: float | int, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(float(value) / multiple) * multiple)


def _round_up_power_of_two(value: float | int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(math.ceil(float(value))) - 1).bit_length()


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_numeric(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [
        float(value)
        for value in values
        if value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not numbers:
        return {"count": 0}

    sorted_values = sorted(numbers)
    summary: dict[str, Any] = {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean(sorted_values),
        "std": pstdev(sorted_values) if len(sorted_values) > 1 else 0.0,
    }
    for percentile in SUMMARY_PERCENTILES:
        label = str(percentile).replace(".", "_")
        summary[f"p{label}"] = _percentile(sorted_values, percentile)
    return summary


def _tokenizer_input_ids(processor: LlavaAnythingProcessor, text: str) -> list[int]:
    encoded = processor.tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return list(encoded["input_ids"])


def _rendered_image_marker_count(processor: LlavaAnythingProcessor, text: str) -> int:
    return text.count(processor.image_token)


def _expand_text_for_training(
    processor: LlavaAnythingProcessor,
    text: str,
    image_size: list[int] | None,
) -> str:
    if image_size is not None:
        return processor._expand_image_tokens(text, [image_size])
    return processor._expand_image_tokens(text)


def _record_id(record: dict[str, Any], index: int) -> str:
    value = record.get("id")
    return str(value) if value is not None else str(index)


def _record_turn_count(record: dict[str, Any]) -> int:
    conversations = record.get("conversations")
    return len(conversations) if isinstance(conversations, list) else 0


def _declared_image_count(record: dict[str, Any]) -> int:
    count = 1 if record.get("image") else 0
    images = record.get("images")
    if isinstance(images, list):
        count += len([item for item in images if item])
    return count


def _training_image_path(record: dict[str, Any], image_folder: Path) -> Path | None:
    image_name = record.get("image")
    if not image_name:
        return None
    return image_folder / str(image_name)


def _read_image_size(path: Path, timeout_seconds: float | None = None) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read image sizes.") from exc

    def timeout_handler(signum: int, frame: Any) -> None:
        raise ImageOpenTimeoutError(f"timed out after {timeout_seconds:g}s while reading {path}")

    previous_handler: Any = None
    use_timeout = timeout_seconds is not None and timeout_seconds > 0
    if use_timeout:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        with Image.open(path) as image:
            width, height = image.size
        return width, height
    finally:
        if use_timeout:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)


def _image_aspect_ratio(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return math.inf
    return max(width / height, height / width)


def _empty_row(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_index": index,
        "record_id": _record_id(record, index),
        "status": "skipped",
        "skip_reason": "",
        "usable_without_context_cap": False,
        "usable_for_configured_training": False,
        "has_training_image": bool(record.get("image")),
        "declared_image_count": _declared_image_count(record),
        "rendered_image_markers": None,
        "conversation_turns": _record_turn_count(record),
        "image_path": "",
        "image_width": None,
        "image_height": None,
        "image_pixels": None,
        "image_aspect_ratio": None,
        "image_tokens": 0,
        "full_tokens": None,
        "text_tokens": None,
        "prompt_tokens": None,
        "prompt_text_tokens": None,
        "target_tokens": None,
        "target_text_tokens": None,
        "image_tokens_in_prompt": None,
        "image_tokens_in_target": None,
        "fits_model_max_length": None,
        "needs_truncation_at_model_max_length": False,
        "length_over_model_max_length": None,
        "truncated_total_tokens": None,
    }


def _mark_skipped(row: dict[str, Any], reason: str) -> dict[str, Any]:
    row["status"] = "skipped"
    row["skip_reason"] = reason
    row["usable_without_context_cap"] = False
    row["usable_for_configured_training"] = False
    return row


def _context_fit(
    *,
    full_tokens: int,
    image_tokens: int,
    target_text_tokens: int,
    context_length: int | None,
) -> dict[str, Any]:
    if context_length is None:
        return {
            "fits": True,
            "needs_truncation": False,
            "trainable": True,
            "reason": "",
            "truncated_total_tokens": full_tokens,
        }

    fits = full_tokens <= context_length
    if fits:
        return {
            "fits": True,
            "needs_truncation": False,
            "trainable": True,
            "reason": "",
            "truncated_total_tokens": full_tokens,
        }

    if image_tokens >= context_length:
        return {
            "fits": False,
            "needs_truncation": True,
            "trainable": False,
            "reason": "image_tokens_exceed_model_max_length",
            "truncated_total_tokens": None,
        }

    text_budget = context_length - image_tokens
    if target_text_tokens <= 0 or text_budget <= 0:
        return {
            "fits": False,
            "needs_truncation": True,
            "trainable": False,
            "reason": "no_target_tokens_fit_model_max_length",
            "truncated_total_tokens": None,
        }

    return {
        "fits": False,
        "needs_truncation": True,
        "trainable": True,
        "reason": "",
        "truncated_total_tokens": context_length,
    }


def _image_constraint_failure(
    processor: LlavaAnythingProcessor,
    width: int,
    height: int,
    options: dict[str, Any],
) -> str | None:
    min_width = options.get("min_image_width")
    if min_width is not None and width < int(min_width):
        return f"image_width_below_min:{width}<{int(min_width)}"

    min_height = options.get("min_image_height")
    if min_height is not None and height < int(min_height):
        return f"image_height_below_min:{height}<{int(min_height)}"

    max_aspect_ratio = options.get("max_image_aspect_ratio")
    aspect_ratio = _image_aspect_ratio(width, height)
    if max_aspect_ratio is not None and aspect_ratio > float(max_aspect_ratio):
        return f"image_aspect_ratio_above_max:{aspect_ratio:.6g}>{float(max_aspect_ratio):.6g}"

    max_image_tokens = options.get("max_image_tokens")
    if max_image_tokens is not None:
        image_tokens = processor._num_image_tokens([height, width])
        if image_tokens > int(max_image_tokens):
            return f"image_tokens_above_max:{image_tokens}>{int(max_image_tokens)}"

    return None


def analyze_record(
    index: int,
    record: dict[str, Any],
    processor: LlavaAnythingProcessor,
    options: dict[str, Any],
) -> dict[str, Any]:
    row = _empty_row(index, record)
    image_folder = Path(options["image_folder"])
    model_max_length = options.get("model_max_length")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    try:
        prefix, assistant_text = _preview_record(
            processor,
            record,
            options.get("system_prompt"),
        )
    except Exception as exc:
        return _mark_skipped(row, f"conversation_error:{exc}")

    eos = processor.tokenizer.eos_token or ""
    full_text = f"{prefix}{assistant_text}{eos}"
    marker_count = _rendered_image_marker_count(processor, full_text)
    row["rendered_image_markers"] = marker_count

    image_path = _training_image_path(record, image_folder)
    has_training_image = image_path is not None
    row["has_training_image"] = has_training_image
    if image_path is not None:
        row["image_path"] = str(image_path)

    if options.get("require_image") and not has_training_image:
        return _mark_skipped(row, "text_only_record_filtered_by_require_image")

    width: int | None = None
    height: int | None = None
    image_size: list[int] | None = None
    expected_image_tokens = 0
    training_filter_reason: str | None = None

    if has_training_image:
        if not image_path.is_file():
            reason = "missing_image"
            if not options.get("available_images_only", True):
                reason = "missing_image_would_fail_training_load"
            return _mark_skipped(row, reason)

        try:
            width, height = _read_image_size(
                image_path,
                timeout_seconds=options.get("image_open_timeout_seconds"),
            )
        except ImageOpenTimeoutError as exc:
            return _mark_skipped(row, f"image_open_timeout:{exc}")
        except Exception as exc:
            return _mark_skipped(row, f"image_open_error:{exc}")

        image_size = [height, width]
        row["image_width"] = width
        row["image_height"] = height
        row["image_pixels"] = width * height
        row["image_aspect_ratio"] = _image_aspect_ratio(width, height)

        try:
            expected_image_tokens = int(processor._num_image_tokens(image_size))
        except Exception as exc:
            return _mark_skipped(row, f"image_token_count_error:{exc}")
        row["image_tokens"] = expected_image_tokens

        constraint_failure = _image_constraint_failure(processor, width, height, options)
        if constraint_failure is not None:
            training_filter_reason = constraint_failure

        if marker_count != 1:
            return _mark_skipped(
                row,
                f"image_token_marker_mismatch:markers={marker_count},training_images=1",
            )
    elif marker_count:
        return _mark_skipped(row, "text_only_record_contains_image_token")

    try:
        if has_training_image:
            expanded_full = _expand_text_for_training(processor, full_text, image_size)
            expanded_prefix = _expand_text_for_training(processor, prefix, image_size)
        else:
            expanded_full = processor._expand_image_tokens(full_text)
            expanded_prefix = processor._expand_image_tokens(prefix)
        full_ids = _tokenizer_input_ids(processor, expanded_full)
        prefix_ids = _tokenizer_input_ids(processor, expanded_prefix)
    except Exception as exc:
        return _mark_skipped(row, f"tokenization_error:{exc}")

    prefix_length = len(prefix_ids)
    full_tokens = len(full_ids)
    image_tokens = int(sum(1 for token_id in full_ids if token_id == image_token_id))
    prompt_image_tokens = int(sum(1 for token_id in full_ids[:prefix_length] if token_id == image_token_id))
    target_image_tokens = int(sum(1 for token_id in full_ids[prefix_length:] if token_id == image_token_id))
    target_text_tokens = int(sum(1 for token_id in full_ids[prefix_length:] if token_id != image_token_id))
    prompt_text_tokens = prefix_length - prompt_image_tokens

    row.update(
        {
            "image_tokens": image_tokens,
            "full_tokens": full_tokens,
            "text_tokens": full_tokens - image_tokens,
            "prompt_tokens": prefix_length,
            "prompt_text_tokens": prompt_text_tokens,
            "target_tokens": full_tokens - prefix_length,
            "target_text_tokens": target_text_tokens,
            "image_tokens_in_prompt": prompt_image_tokens,
            "image_tokens_in_target": target_image_tokens,
        }
    )

    if has_training_image and image_tokens != expected_image_tokens:
        return _mark_skipped(
            row,
            f"expanded_image_token_mismatch:actual={image_tokens},expected={expected_image_tokens}",
        )

    if target_text_tokens <= 0:
        return _mark_skipped(row, "no_supervised_target_tokens")

    fit = _context_fit(
        full_tokens=full_tokens,
        image_tokens=image_tokens,
        target_text_tokens=target_text_tokens,
        context_length=model_max_length,
    )
    row["fits_model_max_length"] = fit["fits"] if model_max_length is not None else None
    row["needs_truncation_at_model_max_length"] = fit["needs_truncation"]
    row["length_over_model_max_length"] = (
        max(0, full_tokens - int(model_max_length)) if model_max_length is not None else None
    )
    row["truncated_total_tokens"] = fit["truncated_total_tokens"]

    if training_filter_reason is not None:
        row["status"] = "skipped"
        row["skip_reason"] = training_filter_reason
        row["usable_without_context_cap"] = False
        row["usable_for_configured_training"] = False
        return row

    row["usable_without_context_cap"] = True
    row["usable_for_configured_training"] = bool(fit["trainable"])

    if fit["trainable"]:
        row["status"] = "usable_truncated" if fit["needs_truncation"] else "usable"
        row["skip_reason"] = ""
    else:
        row["status"] = "skipped_by_configured_context"
        row["skip_reason"] = fit["reason"]
    return row


def _worker_initializer(options: dict[str, Any]) -> None:
    global _WORKER_OPTIONS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_OPTIONS = options


def _analyze_record_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _WORKER_PROCESSOR is None:
        raise RuntimeError("worker processor was not initialized")
    index, record = item
    return analyze_record(index, record, _WORKER_PROCESSOR, _WORKER_OPTIONS)


def _analyze_record_chunk_worker(chunk: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    if _WORKER_PROCESSOR is None:
        raise RuntimeError("worker processor was not initialized")
    return [
        analyze_record(index, record, _WORKER_PROCESSOR, _WORKER_OPTIONS)
        for index, record in chunk
    ]


def load_processor_from_training_yaml(training_data: dict[str, Any]) -> LlavaAnythingProcessor:
    _load_runtime_imports()

    model_yaml = training_data.get("model_yaml")
    model_checkpoint = training_data.get("model_checkpoint")
    if bool(model_yaml) == bool(model_checkpoint):
        raise ValueError("Exactly one of model_yaml or model_checkpoint is required.")

    if model_checkpoint:
        return LlavaAnythingProcessor.from_pretrained(Path(model_checkpoint))

    model_data = load_yaml(model_yaml)
    config = config_from_yaml_dict(model_data)
    return processor_from_yaml_dict(model_data, config)


def _records_for_analysis(data_path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    _load_runtime_imports()

    return _load_json_records(data_path, max_samples=max_samples)


def iter_json_records(path: Path, max_samples: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield parsed records without retaining the whole dataset in memory."""

    path = Path(path)
    yielded = 0
    if path.suffix == ".jsonl":
        print(f"Streaming JSONL records from {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if max_samples is not None and yielded >= max_samples:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object record at {path}:{line_no}")
                yield yielded, record
                yielded += 1
        return

    print(f"Loading JSON records from {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    for record in records:
        if max_samples is not None and yielded >= max_samples:
            break
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object record at index {yielded} in {path}")
        yield yielded, record
        yielded += 1


def iter_record_chunks(
    path: Path,
    max_samples: int | None,
    chunk_size: int,
) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    """Yield bounded record chunks for multiprocessing."""

    chunk: list[tuple[int, dict[str, Any]]] = []
    for item in iter_json_records(path, max_samples=max_samples):
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _default_output_dir(training_yaml: Path, training_data: dict[str, Any]) -> Path:
    analysis_section = training_data.get("analysis", {}) or {}
    if isinstance(analysis_section, dict) and analysis_section.get("output_dir"):
        return Path(analysis_section["output_dir"])
    training_section = training_data.get("training", {}) or {}
    if isinstance(training_section, dict) and training_section.get("output_dir"):
        return Path(training_section["output_dir"]) / "dataset_analysis"
    return Path("dataset_analysis") / training_yaml.stem


def _analysis_options(
    training_data: dict[str, Any],
    processor: LlavaAnythingProcessor,
    *,
    model_max_length_override: int | None,
    respect_max_image_tokens: bool,
    image_open_timeout_seconds: float | None,
) -> dict[str, Any]:
    _load_runtime_imports()

    data_section = training_data.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("data must be a mapping.")
    training_section = training_data.get("training", {}) or {}
    if not isinstance(training_section, dict):
        raise ValueError("training must be a mapping when provided.")

    model_max_length_raw = (
        model_max_length_override
        if model_max_length_override is not None
        else training_section.get("model_max_length")
    )
    model_max_length = _resolve_model_max_length(processor, model_max_length_raw)

    return {
        "image_folder": str(Path(data_section["image_folder"])),
        "available_images_only": bool(data_section.get("available_images_only", True)),
        "require_image": bool(data_section.get("require_image", False)),
        "min_image_width": data_section.get("min_image_width"),
        "min_image_height": data_section.get("min_image_height"),
        "max_image_aspect_ratio": data_section.get("max_image_aspect_ratio"),
        "max_image_tokens": data_section.get("max_image_tokens") if respect_max_image_tokens else None,
        "configured_max_image_tokens": data_section.get("max_image_tokens"),
        "respect_max_image_tokens": bool(respect_max_image_tokens),
        "image_open_timeout_seconds": image_open_timeout_seconds,
        "system_prompt": data_section.get("system_prompt"),
        "model_max_length": model_max_length,
    }


def analyze_records(
    records: list[dict[str, Any]],
    processor: LlavaAnythingProcessor,
    options: dict[str, Any],
    *,
    num_workers: int,
    chunksize: int,
) -> list[dict[str, Any]]:
    global _WORKER_PROCESSOR, _WORKER_OPTIONS

    if num_workers <= 1 or len(records) <= 1:
        _WORKER_PROCESSOR = processor
        _WORKER_OPTIONS = options
        return [
            analyze_record(index, record, processor, options)
            for index, record in tqdm(
                list(enumerate(records)),
                total=len(records),
                desc="Analyzing records",
                unit="records",
                mininterval=5.0,
            )
        ]

    start_methods = mp.get_all_start_methods()
    if "fork" not in start_methods:
        warnings.warn(
            "Process-based analysis needs the fork start method to share the loaded processor. "
            "Falling back to one worker.",
            UserWarning,
            stacklevel=2,
        )
        return analyze_records(
            records,
            processor,
            options,
            num_workers=1,
            chunksize=chunksize,
        )

    _WORKER_PROCESSOR = processor
    _WORKER_OPTIONS = options
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(options,),
    ) as executor:
        iterator = executor.map(
            _analyze_record_worker,
            enumerate(records),
            chunksize=max(1, chunksize),
        )
        return list(
            tqdm(
                iterator,
                total=len(records),
                desc=f"Analyzing records with {num_workers} workers",
                unit="records",
                mininterval=5.0,
            )
        )


def analyze_record_chunks(
    data_path: Path,
    processor: LlavaAnythingProcessor,
    options: dict[str, Any],
    *,
    max_samples: int | None,
    num_workers: int,
    chunksize: int,
    max_pending_chunks: int | None = None,
) -> Iterable[list[dict[str, Any]]]:
    """Analyze records from disk in bounded chunks without retaining all rows."""

    global _WORKER_PROCESSOR, _WORKER_OPTIONS

    chunk_size = max(1, int(chunksize))
    total = max_samples if max_samples is not None else None
    desc = "Analyzing streamed records" if num_workers <= 1 else f"Analyzing streamed records with {num_workers} workers"

    if num_workers <= 1:
        _WORKER_PROCESSOR = processor
        _WORKER_OPTIONS = options
        with tqdm(total=total, desc=desc, unit="records", mininterval=5.0) as progress:
            for chunk in iter_record_chunks(data_path, max_samples, chunk_size):
                rows = [
                    analyze_record(index, record, processor, options)
                    for index, record in chunk
                ]
                progress.update(len(rows))
                yield rows
        return

    start_methods = mp.get_all_start_methods()
    if "fork" not in start_methods:
        warnings.warn(
            "Process-based analysis needs the fork start method to share the loaded processor. "
            "Falling back to one worker.",
            UserWarning,
            stacklevel=2,
        )
        yield from analyze_record_chunks(
            data_path,
            processor,
            options,
            max_samples=max_samples,
            num_workers=1,
            chunksize=chunk_size,
        )
        return

    _WORKER_PROCESSOR = processor
    _WORKER_OPTIONS = options
    pending_limit = max_pending_chunks or max(1, num_workers * 2)
    chunk_iter = iter(iter_record_chunks(data_path, max_samples, chunk_size))
    context = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(options,),
    ) as executor, tqdm(total=total, desc=desc, unit="records", mininterval=5.0) as progress:
        pending = set()
        exhausted = False

        def submit_until_full() -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < pending_limit:
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(_analyze_record_chunk_worker, chunk))

        submit_until_full()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                rows = future.result()
                progress.update(len(rows))
                yield rows
            submit_until_full()


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(values).most_common()}


def _base_usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("usable_without_context_cap")]


def _configured_usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("usable_for_configured_training")]


def _is_broken_image_row(row: dict[str, Any]) -> bool:
    reason = str(row.get("skip_reason") or "")
    return (
        reason == "missing_image"
        or reason == "missing_image_would_fail_training_load"
        or reason.startswith("image_open_timeout:")
        or reason.startswith("image_open_error:")
    )


def _broken_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_broken_image_row(row)]


def _analyzable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("full_tokens") is not None
        and row.get("target_text_tokens") is not None
        and int(row["target_text_tokens"]) > 0
    ]


def _context_coverage(
    rows: list[dict[str, Any]],
    context_lengths: list[int],
) -> dict[str, dict[str, Any]]:
    total = len(rows)
    coverage: dict[str, dict[str, Any]] = {}
    for context_length in sorted(set(context_lengths)):
        if context_length <= 0:
            continue
        untruncated = 0
        trainable = 0
        truncated = 0
        context_skipped = 0
        for row in rows:
            fit = _context_fit(
                full_tokens=int(row["full_tokens"]),
                image_tokens=int(row["image_tokens"]),
                target_text_tokens=int(row["target_text_tokens"]),
                context_length=context_length,
            )
            if fit["fits"]:
                untruncated += 1
            if fit["trainable"]:
                trainable += 1
                if fit["needs_truncation"]:
                    truncated += 1
            else:
                context_skipped += 1

        coverage[str(context_length)] = {
            "records": total,
            "untruncated_count": untruncated,
            "untruncated_fraction": (untruncated / total) if total else None,
            "trainable_with_truncation_count": trainable,
            "trainable_with_truncation_fraction": (trainable / total) if total else None,
            "would_truncate_count": truncated,
            "skipped_by_context_count": context_skipped,
        }
    return coverage


def _context_recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(float(row["full_tokens"]) for row in rows if row.get("full_tokens") is not None)
    recommendations: dict[str, Any] = {}
    for percentile in SUMMARY_PERCENTILES:
        value = _percentile(lengths, percentile)
        if value is None:
            continue
        label = str(percentile).replace(".", "_")
        recommendations[f"p{label}"] = {
            "tokens": value,
            "rounded_to_256": _round_up(value, 256),
            "rounded_to_512": _round_up(value, 512),
            "rounded_to_power_of_two": _round_up_power_of_two(value),
        }
    return recommendations


def _context_recommendations_from_values(values: Iterable[float]) -> dict[str, Any]:
    lengths = sorted(float(value) for value in values)
    recommendations: dict[str, Any] = {}
    for percentile in SUMMARY_PERCENTILES:
        value = _percentile(lengths, percentile)
        if value is None:
            continue
        label = str(percentile).replace(".", "_")
        recommendations[f"p{label}"] = {
            "tokens": value,
            "rounded_to_256": _round_up(value, 256),
            "rounded_to_512": _round_up(value, 512),
            "rounded_to_power_of_two": _round_up_power_of_two(value),
        }
    return recommendations


class AnalysisAccumulator:
    """Accumulate dataset statistics without storing full record rows."""

    def __init__(self, context_lengths: list[int], model_max_length: int | None = None) -> None:
        evaluated_context_lengths = list(context_lengths)
        if model_max_length is not None:
            evaluated_context_lengths.append(int(model_max_length))
        evaluated_context_lengths.extend(DEFAULT_CONTEXT_LENGTHS)
        self.context_lengths = sorted({int(value) for value in evaluated_context_lengths if int(value) > 0})
        self.counts = Counter()
        self.status_counts = Counter()
        self.skip_reason_counts = Counter()
        self.images_per_record_counts = Counter()
        self.length_values: dict[str, list[float]] = {
            "full_tokens": [],
            "text_tokens": [],
            "image_tokens": [],
            "prompt_text_tokens": [],
            "target_text_tokens": [],
            "length_over_configured_model_max": [],
        }
        self.training_length_values: dict[str, list[float]] = {
            key: [] for key in self.length_values
        }
        self.image_values: dict[str, list[float]] = {
            "image_width": [],
            "image_height": [],
            "image_pixels": [],
            "image_aspect_ratio": [],
            "image_tokens": [],
        }
        self.coverage_all = self._empty_coverage()
        self.coverage_training = self._empty_coverage()
        self.image_size_plot_widths: list[float] = []
        self.image_size_plot_heights: list[float] = []
        self.image_size_plot_tokens: list[float] = []
        self.max_image_size_plot_points = 200_000

    def _empty_coverage(self) -> dict[int, Counter]:
        return {
            length: Counter(
                {
                    "records": 0,
                    "untruncated_count": 0,
                    "trainable_with_truncation_count": 0,
                    "would_truncate_count": 0,
                    "skipped_by_context_count": 0,
                }
            )
            for length in self.context_lengths
        }

    @staticmethod
    def _add_value(values: dict[str, list[float]], key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[key].append(float(value))

    @staticmethod
    def _is_analyzable(row: dict[str, Any]) -> bool:
        return (
            row.get("full_tokens") is not None
            and row.get("target_text_tokens") is not None
            and int(row["target_text_tokens"]) > 0
        )

    def _add_length_values(self, target: dict[str, list[float]], row: dict[str, Any]) -> None:
        for key in (
            "full_tokens",
            "text_tokens",
            "image_tokens",
            "prompt_text_tokens",
            "target_text_tokens",
            "length_over_model_max_length",
        ):
            target_key = "length_over_configured_model_max" if key == "length_over_model_max_length" else key
            self._add_value(target, target_key, row.get(key))

    def _update_coverage(self, coverage: dict[int, Counter], row: dict[str, Any]) -> None:
        for context_length in self.context_lengths:
            fit = _context_fit(
                full_tokens=int(row["full_tokens"]),
                image_tokens=int(row["image_tokens"]),
                target_text_tokens=int(row["target_text_tokens"]),
                context_length=context_length,
            )
            item = coverage[context_length]
            item["records"] += 1
            if fit["fits"]:
                item["untruncated_count"] += 1
            if fit["trainable"]:
                item["trainable_with_truncation_count"] += 1
                if fit["needs_truncation"]:
                    item["would_truncate_count"] += 1
            else:
                item["skipped_by_context_count"] += 1

    @staticmethod
    def _format_coverage(coverage: dict[int, Counter]) -> dict[str, dict[str, Any]]:
        formatted: dict[str, dict[str, Any]] = {}
        for context_length, item in sorted(coverage.items()):
            total = int(item["records"])
            formatted[str(context_length)] = {
                "records": total,
                "untruncated_count": int(item["untruncated_count"]),
                "untruncated_fraction": (item["untruncated_count"] / total) if total else None,
                "trainable_with_truncation_count": int(item["trainable_with_truncation_count"]),
                "trainable_with_truncation_fraction": (
                    item["trainable_with_truncation_count"] / total
                )
                if total
                else None,
                "would_truncate_count": int(item["would_truncate_count"]),
                "skipped_by_context_count": int(item["skipped_by_context_count"]),
            }
        return formatted

    def update(self, row: dict[str, Any]) -> None:
        self.counts["records_loaded"] += 1
        if row.get("has_training_image"):
            self.counts["records_with_training_image"] += 1
        else:
            self.counts["text_only_records"] += 1
        if row.get("needs_truncation_at_model_max_length"):
            self.counts["records_needing_truncation_at_configured_length"] += 1
        if row.get("usable_without_context_cap"):
            self.counts["usable_without_context_cap"] += 1
        if row.get("usable_for_configured_training"):
            self.counts["usable_for_configured_training"] += 1
        if _is_broken_image_row(row):
            self.counts["broken_or_missing_image_records"] += 1

        self.status_counts.update([row.get("status")])
        if row.get("skip_reason"):
            self.skip_reason_counts.update([row.get("skip_reason")])
        self.images_per_record_counts.update([row.get("declared_image_count")])

        for key in ("image_width", "image_height", "image_pixels", "image_aspect_ratio"):
            self._add_value(self.image_values, key, row.get(key))
        if row.get("has_training_image") and row.get("image_width") is not None:
            self._add_value(self.image_values, "image_tokens", row.get("image_tokens"))
            if len(self.image_size_plot_widths) < self.max_image_size_plot_points:
                self.image_size_plot_widths.append(float(row["image_width"]))
                self.image_size_plot_heights.append(float(row["image_height"]))
                self.image_size_plot_tokens.append(float(row.get("image_tokens", 0)))

        if self._is_analyzable(row):
            self.counts["analyzable_records"] += 1
            self._add_length_values(self.length_values, row)
            self._update_coverage(self.coverage_all, row)

        if row.get("usable_without_context_cap"):
            self._add_length_values(self.training_length_values, row)
            self._update_coverage(self.coverage_training, row)

    def to_summary(
        self,
        *,
        training_yaml: Path,
        data_path: Path,
        image_folder: Path,
        processor: LlavaAnythingProcessor,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        records_loaded = int(self.counts["records_loaded"])
        configured_usable = int(self.counts["usable_for_configured_training"])
        return {
            "input": {
                "training_yaml": str(training_yaml),
                "data_path": str(data_path),
                "image_folder": str(image_folder),
            },
            "processor": {
                "image_token": processor.image_token,
                "image_mode": getattr(processor, "image_mode", None),
                "image_seq_length": getattr(processor, "image_seq_length", None),
                "patch_size": getattr(processor, "patch_size", None),
                "image_grid_pinpoints": getattr(processor, "image_grid_pinpoints", None),
                "vision_feature_select_strategy": getattr(processor, "vision_feature_select_strategy", None),
                "num_additional_image_tokens": getattr(processor, "num_additional_image_tokens", None),
                "tokenizer_model_max_length": getattr(processor.tokenizer, "model_max_length", None),
                "configured_model_max_length": options.get("model_max_length"),
                "configured_max_image_tokens": options.get("configured_max_image_tokens"),
                "respect_max_image_tokens": options.get("respect_max_image_tokens"),
                "image_open_timeout_seconds": options.get("image_open_timeout_seconds"),
            },
            "counts": {
                "records_loaded": records_loaded,
                "analyzable_records": int(self.counts["analyzable_records"]),
                "usable_without_context_cap": int(self.counts["usable_without_context_cap"]),
                "usable_for_configured_training": configured_usable,
                "skipped": records_loaded - configured_usable,
                "text_only_records": int(self.counts["text_only_records"]),
                "records_with_training_image": int(self.counts["records_with_training_image"]),
                "broken_or_missing_image_records": int(self.counts["broken_or_missing_image_records"]),
                "records_needing_truncation_at_configured_length": int(
                    self.counts["records_needing_truncation_at_configured_length"]
                ),
            },
            "status_counts": _counter_dict(self.status_counts.elements()),
            "skip_reason_counts": _counter_dict(self.skip_reason_counts.elements()),
            "images_per_record_counts": _counter_dict(self.images_per_record_counts.elements()),
            "length_statistics": {
                key: summarize_numeric(values) for key, values in self.length_values.items()
            },
            "training_length_statistics": {
                key: summarize_numeric(values) for key, values in self.training_length_values.items()
            },
            "image_statistics": {
                key: summarize_numeric(values) for key, values in self.image_values.items()
            },
            "context_length_guidance": {
                "recommendations_for_all_analyzable_records_no_truncation": _context_recommendations_from_values(
                    self.length_values["full_tokens"]
                ),
                "recommendations_for_training_records_no_truncation": _context_recommendations_from_values(
                    self.training_length_values["full_tokens"]
                ),
                "coverage_at_context_lengths_all_analyzable": self._format_coverage(self.coverage_all),
                "coverage_at_context_lengths_training_records": self._format_coverage(self.coverage_training),
            },
        }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    training_yaml: Path,
    data_path: Path,
    image_folder: Path,
    processor: LlavaAnythingProcessor,
    options: dict[str, Any],
    context_lengths: list[int],
) -> dict[str, Any]:
    base_rows = _base_usable_rows(rows)
    configured_rows = _configured_usable_rows(rows)
    analyzable_rows = _analyzable_rows(rows)
    broken_image_rows = _broken_image_rows(rows)
    model_max_length = options.get("model_max_length")
    evaluated_context_lengths = list(context_lengths)
    if model_max_length is not None:
        evaluated_context_lengths.append(int(model_max_length))
    evaluated_context_lengths.extend(DEFAULT_CONTEXT_LENGTHS)

    return {
        "input": {
            "training_yaml": str(training_yaml),
            "data_path": str(data_path),
            "image_folder": str(image_folder),
        },
        "processor": {
            "image_token": processor.image_token,
            "image_mode": getattr(processor, "image_mode", None),
            "image_seq_length": getattr(processor, "image_seq_length", None),
            "patch_size": getattr(processor, "patch_size", None),
            "image_grid_pinpoints": getattr(processor, "image_grid_pinpoints", None),
            "vision_feature_select_strategy": getattr(processor, "vision_feature_select_strategy", None),
            "num_additional_image_tokens": getattr(processor, "num_additional_image_tokens", None),
            "tokenizer_model_max_length": getattr(processor.tokenizer, "model_max_length", None),
            "configured_model_max_length": model_max_length,
            "configured_max_image_tokens": options.get("configured_max_image_tokens"),
            "respect_max_image_tokens": options.get("respect_max_image_tokens"),
            "image_open_timeout_seconds": options.get("image_open_timeout_seconds"),
        },
        "counts": {
            "records_loaded": len(rows),
            "analyzable_records": len(analyzable_rows),
            "usable_without_context_cap": len(base_rows),
            "usable_for_configured_training": len(configured_rows),
            "skipped": len(rows) - len(configured_rows),
            "text_only_records": sum(1 for row in rows if not row.get("has_training_image")),
            "records_with_training_image": sum(1 for row in rows if row.get("has_training_image")),
            "broken_or_missing_image_records": len(broken_image_rows),
            "records_needing_truncation_at_configured_length": sum(
                1 for row in rows if row.get("needs_truncation_at_model_max_length")
            ),
        },
        "status_counts": _counter_dict(row.get("status") for row in rows),
        "skip_reason_counts": _counter_dict(
            row.get("skip_reason") for row in rows if row.get("skip_reason")
        ),
        "images_per_record_counts": _counter_dict(row.get("declared_image_count") for row in rows),
        "length_statistics": {
            "full_tokens": summarize_numeric(row.get("full_tokens") for row in analyzable_rows),
            "text_tokens": summarize_numeric(row.get("text_tokens") for row in analyzable_rows),
            "image_tokens": summarize_numeric(row.get("image_tokens") for row in analyzable_rows),
            "prompt_text_tokens": summarize_numeric(row.get("prompt_text_tokens") for row in analyzable_rows),
            "target_text_tokens": summarize_numeric(row.get("target_text_tokens") for row in analyzable_rows),
            "length_over_configured_model_max": summarize_numeric(
                row.get("length_over_model_max_length")
                for row in analyzable_rows
                if row.get("length_over_model_max_length") is not None
            ),
        },
        "training_length_statistics": {
            "full_tokens": summarize_numeric(row.get("full_tokens") for row in base_rows),
            "text_tokens": summarize_numeric(row.get("text_tokens") for row in base_rows),
            "image_tokens": summarize_numeric(row.get("image_tokens") for row in base_rows),
            "prompt_text_tokens": summarize_numeric(row.get("prompt_text_tokens") for row in base_rows),
            "target_text_tokens": summarize_numeric(row.get("target_text_tokens") for row in base_rows),
            "length_over_configured_model_max": summarize_numeric(
                row.get("length_over_model_max_length")
                for row in base_rows
                if row.get("length_over_model_max_length") is not None
            ),
        },
        "image_statistics": {
            "image_width": summarize_numeric(row.get("image_width") for row in rows),
            "image_height": summarize_numeric(row.get("image_height") for row in rows),
            "image_pixels": summarize_numeric(row.get("image_pixels") for row in rows),
            "image_aspect_ratio": summarize_numeric(row.get("image_aspect_ratio") for row in rows),
            "image_tokens": summarize_numeric(
                row.get("image_tokens")
                for row in rows
                if row.get("has_training_image") and row.get("image_width") is not None
            ),
        },
        "context_length_guidance": {
            "recommendations_for_all_analyzable_records_no_truncation": _context_recommendations(analyzable_rows),
            "recommendations_for_training_records_no_truncation": _context_recommendations(base_rows),
            "coverage_at_context_lengths_all_analyzable": _context_coverage(
                analyzable_rows,
                evaluated_context_lengths,
            ),
            "coverage_at_context_lengths_training_records": _context_coverage(
                base_rows,
                evaluated_context_lengths,
            ),
        },
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return value


def write_records_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in RECORD_FIELDS})


def write_broken_images_csv(rows: list[dict[str, Any]], path: Path) -> int:
    broken_rows = _broken_image_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BROKEN_IMAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in broken_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in BROKEN_IMAGE_FIELDS})
    return len(broken_rows)


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_stat_line(name: str, stats: dict[str, Any]) -> str:
    if not stats or not stats.get("count"):
        return f"- {name}: no values"
    return (
        f"- {name}: mean={stats['mean']:.1f}, p90={stats.get('p90', 0):.0f}, "
        f"p95={stats.get('p95', 0):.0f}, p99={stats.get('p99', 0):.0f}, max={stats['max']:.0f}"
    )


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    counts = summary["counts"]
    length_stats = summary["length_statistics"]
    guidance = summary["context_length_guidance"]
    all_recommendations = guidance["recommendations_for_all_analyzable_records_no_truncation"]
    training_recommendations = guidance["recommendations_for_training_records_no_truncation"]
    model_max_length = summary["processor"].get("configured_model_max_length")

    lines = [
        "# Dataset Analysis",
        "",
        f"- Records loaded: {counts['records_loaded']}",
        f"- Analyzable records: {counts['analyzable_records']}",
        f"- Usable before context cap: {counts['usable_without_context_cap']}",
        f"- Usable at configured context: {counts['usable_for_configured_training']}",
        f"- Broken or missing image records: {counts['broken_or_missing_image_records']}",
        f"- Configured model_max_length: {model_max_length if model_max_length is not None else 'not set'}",
        (
            f"- Configured max_image_tokens: {summary['processor'].get('configured_max_image_tokens')} "
            f"({'applied' if summary['processor'].get('respect_max_image_tokens') else 'ignored for raw analysis'})"
        ),
        "",
        "## Token Lengths",
        "",
        _format_stat_line("full tokens", length_stats["full_tokens"]),
        _format_stat_line("text tokens", length_stats["text_tokens"]),
        _format_stat_line("image tokens", length_stats["image_tokens"]),
        _format_stat_line("target text tokens", length_stats["target_text_tokens"]),
        "",
        "## Context Length Recommendations",
        "",
        "All analyzable records:",
        "",
    ]
    for key in ("p90", "p95", "p99", "p99_5", "p100"):
        if key not in all_recommendations:
            continue
        item = all_recommendations[key]
        lines.append(
            f"- {key}: {item['tokens']:.0f} tokens "
            f"(round to 256: {item['rounded_to_256']}, "
            f"power of two: {item['rounded_to_power_of_two']})"
        )

    lines.extend(["", "Training-filtered records:", ""])
    wrote_training_recommendation = False
    for key in ("p90", "p95", "p99", "p99_5", "p100"):
        if key not in training_recommendations:
            continue
        item = training_recommendations[key]
        wrote_training_recommendation = True
        lines.append(
            f"- {key}: {item['tokens']:.0f} tokens "
            f"(round to 256: {item['rounded_to_256']}, "
            f"power of two: {item['rounded_to_power_of_two']})"
        )
    if not wrote_training_recommendation:
        lines.append("- no records after training filters")

    lines.extend(["", "## Top Skip Reasons", ""])
    for reason, count in list(summary["skip_reason_counts"].items())[:10]:
        lines.append(f"- {reason}: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_hist(ax: Any, values: list[float], label: str, bins: int = 80) -> None:
    if values:
        ax.hist(values, bins=min(bins, max(5, len(set(values)))), alpha=0.55, label=label)


def write_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    model_max_length: int | None,
    plot_format: str,
) -> list[Path]:
    try:
        mpl_config_dir = output_dir / ".matplotlib"
        xdg_cache_dir = output_dir / ".cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; skipping plot generation.",
            UserWarning,
            stacklevel=2,
        )
        return []

    base_rows = _analyzable_rows(rows)
    written: list[Path] = []

    full_tokens = [float(row["full_tokens"]) for row in base_rows]
    text_tokens = [float(row["text_tokens"]) for row in base_rows]
    image_tokens = [float(row["image_tokens"]) for row in base_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_hist(ax, full_tokens, "full tokens")
    _plot_hist(ax, text_tokens, "text tokens")
    _plot_hist(ax, image_tokens, "image tokens")
    if model_max_length is not None:
        ax.axvline(model_max_length, color="black", linestyle="--", linewidth=1.5, label=f"model_max_length={model_max_length}")
    ax.set_xlabel("tokens")
    ax.set_ylabel("records")
    ax.set_title("Token length distributions")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"token_lengths_hist.{plot_format}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    sorted_lengths = sorted(full_tokens)
    if sorted_lengths:
        y_values = [(index + 1) / len(sorted_lengths) for index in range(len(sorted_lengths))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_lengths, y_values, linewidth=2)
        if model_max_length is not None:
            ax.axvline(model_max_length, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("context length")
        ax.set_ylabel("fraction covered without truncation")
        ax.set_ylim(0, 1.02)
        ax.set_title("Context coverage curve")
        fig.tight_layout()
        path = output_dir / f"context_coverage_ecdf.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    aspect_ratios = [
        float(row["image_aspect_ratio"])
        for row in rows
        if row.get("image_aspect_ratio") is not None and math.isfinite(float(row["image_aspect_ratio"]))
    ]
    if aspect_ratios:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_hist(ax, aspect_ratios, "aspect ratio")
        ax.set_xlabel("max(width/height, height/width)")
        ax.set_ylabel("images")
        ax.set_title("Image aspect ratio distribution")
        fig.tight_layout()
        path = output_dir / f"image_aspect_ratios.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    width_height_rows = [
        row
        for row in rows
        if row.get("image_width") is not None and row.get("image_height") is not None
    ]
    if width_height_rows:
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            [row["image_width"] for row in width_height_rows],
            [row["image_height"] for row in width_height_rows],
            c=[row.get("image_tokens", 0) for row in width_height_rows],
            s=10,
            alpha=0.6,
        )
        ax.set_xlabel("image width")
        ax.set_ylabel("image height")
        ax.set_title("Image sizes colored by image-token count")
        fig.colorbar(scatter, ax=ax, label="image tokens")
        fig.tight_layout()
        path = output_dir / f"image_sizes.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    images_per_record = Counter(row.get("declared_image_count", 0) for row in rows)
    if images_per_record:
        fig, ax = plt.subplots(figsize=(8, 5))
        keys = sorted(images_per_record)
        ax.bar([str(key) for key in keys], [images_per_record[key] for key in keys])
        ax.set_xlabel("declared images per record")
        ax.set_ylabel("records")
        ax.set_title("Images per record")
        fig.tight_layout()
        path = output_dir / f"images_per_record.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    return written


def write_plots_from_accumulator(
    accumulator: AnalysisAccumulator,
    output_dir: Path,
    *,
    model_max_length: int | None,
    plot_format: str,
) -> list[Path]:
    try:
        mpl_config_dir = output_dir / ".matplotlib"
        xdg_cache_dir = output_dir / ".cache"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; skipping plot generation.",
            UserWarning,
            stacklevel=2,
        )
        return []

    written: list[Path] = []
    full_tokens = accumulator.length_values["full_tokens"]
    text_tokens = accumulator.length_values["text_tokens"]
    image_tokens = accumulator.length_values["image_tokens"]

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_hist(ax, full_tokens, "full tokens")
    _plot_hist(ax, text_tokens, "text tokens")
    _plot_hist(ax, image_tokens, "image tokens")
    if model_max_length is not None:
        ax.axvline(model_max_length, color="black", linestyle="--", linewidth=1.5, label=f"model_max_length={model_max_length}")
    ax.set_xlabel("tokens")
    ax.set_ylabel("records")
    ax.set_title("Token length distributions")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"token_lengths_hist.{plot_format}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    sorted_lengths = sorted(full_tokens)
    if sorted_lengths:
        y_values = [(index + 1) / len(sorted_lengths) for index in range(len(sorted_lengths))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_lengths, y_values, linewidth=2)
        if model_max_length is not None:
            ax.axvline(model_max_length, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("context length")
        ax.set_ylabel("fraction covered without truncation")
        ax.set_ylim(0, 1.02)
        ax.set_title("Context coverage curve")
        fig.tight_layout()
        path = output_dir / f"context_coverage_ecdf.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    aspect_ratios = accumulator.image_values["image_aspect_ratio"]
    if aspect_ratios:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_hist(ax, aspect_ratios, "aspect ratio")
        ax.set_xlabel("max(width/height, height/width)")
        ax.set_ylabel("images")
        ax.set_title("Image aspect ratio distribution")
        fig.tight_layout()
        path = output_dir / f"image_aspect_ratios.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if accumulator.image_size_plot_widths:
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            accumulator.image_size_plot_widths,
            accumulator.image_size_plot_heights,
            c=accumulator.image_size_plot_tokens,
            s=10,
            alpha=0.45,
        )
        ax.set_xlabel("image width")
        ax.set_ylabel("image height")
        ax.set_title("Image sizes colored by image-token count")
        fig.colorbar(scatter, ax=ax, label="image tokens")
        fig.tight_layout()
        path = output_dir / f"image_sizes.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if accumulator.images_per_record_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        keys = sorted(accumulator.images_per_record_counts)
        ax.bar([str(key) for key in keys], [accumulator.images_per_record_counts[key] for key in keys])
        ax.set_xlabel("declared images per record")
        ax.set_ylabel("records")
        ax.set_title("Images per record")
        fig.tight_layout()
        path = output_dir / f"images_per_record.{plot_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    return written


def run_analysis_from_yaml(
    training_yaml: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_samples: int | None = None,
    num_workers: int | None = None,
    chunksize: int = 64,
    model_max_length: int | None = None,
    context_lengths: list[int] | None = None,
    respect_max_image_tokens: bool = False,
    image_open_timeout_seconds: float | None = DEFAULT_IMAGE_OPEN_TIMEOUT_SECONDS,
    write_plot_files: bool = True,
    plot_format: str = "png",
) -> dict[str, Any]:
    training_yaml = Path(training_yaml)
    _load_runtime_imports()

    training_data = load_yaml(training_yaml)
    data_section = training_data.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("data must be a mapping.")
    if "data_path" not in data_section:
        raise ValueError("data.data_path is required.")
    if "image_folder" not in data_section:
        raise ValueError("data.image_folder is required.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(training_yaml, training_data)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor_from_training_yaml(training_data)
    options = _analysis_options(
        training_data,
        processor,
        model_max_length_override=model_max_length,
        respect_max_image_tokens=respect_max_image_tokens,
        image_open_timeout_seconds=image_open_timeout_seconds,
    )
    effective_max_samples = max_samples if max_samples is not None else data_section.get("max_samples")
    if effective_max_samples is not None:
        effective_max_samples = int(effective_max_samples)
    data_path = Path(data_section["data_path"])
    image_folder = Path(data_section["image_folder"])

    worker_count = (os.cpu_count() or 1) if num_workers is None else int(num_workers)
    worker_count = max(1, int(worker_count))
    accumulator = AnalysisAccumulator(
        context_lengths=context_lengths or [],
        model_max_length=options.get("model_max_length"),
    )

    records_csv_path = resolved_output_dir / "records.csv"
    broken_images_csv_path = resolved_output_dir / "broken_images.csv"
    with records_csv_path.open("w", encoding="utf-8", newline="") as records_handle, broken_images_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as broken_handle:
        records_writer = csv.DictWriter(records_handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        broken_writer = csv.DictWriter(broken_handle, fieldnames=BROKEN_IMAGE_FIELDS, extrasaction="ignore")
        records_writer.writeheader()
        broken_writer.writeheader()
        processed_chunks = 0
        for chunk_rows in analyze_record_chunks(
            data_path,
            processor,
            options,
            max_samples=effective_max_samples,
            num_workers=worker_count,
            chunksize=chunksize,
        ):
            for row in chunk_rows:
                records_writer.writerow({field: _csv_value(row.get(field)) for field in RECORD_FIELDS})
                if _is_broken_image_row(row):
                    broken_writer.writerow({field: _csv_value(row.get(field)) for field in BROKEN_IMAGE_FIELDS})
                accumulator.update(row)
            processed_chunks += 1
            if processed_chunks % 100 == 0:
                records_handle.flush()
                broken_handle.flush()

    summary = accumulator.to_summary(
        training_yaml=training_yaml,
        data_path=data_path,
        image_folder=image_folder,
        processor=processor,
        options=options,
    )
    summary["output_files"] = {
        "records_csv": str(records_csv_path),
        "broken_images_csv": str(broken_images_csv_path),
        "summary_json": str(resolved_output_dir / "summary.json"),
        "report_md": str(resolved_output_dir / "report.md"),
        "plots": [],
    }

    write_json(summary, resolved_output_dir / "summary.json")
    write_markdown_report(summary, resolved_output_dir / "report.md")

    if write_plot_files:
        plots = write_plots_from_accumulator(
            accumulator,
            resolved_output_dir,
            model_max_length=options.get("model_max_length"),
            plot_format=plot_format,
        )
        summary["output_files"]["plots"] = [str(path) for path in plots]
        write_json(summary, resolved_output_dir / "summary.json")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_yaml", type=Path, help="Training YAML accepted by llava-anything-train.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for records.csv, broken_images.csv, summary.json, report.md, and plots. "
            "Defaults to training.output_dir/dataset_analysis."
        ),
    )
    parser.add_argument("--max-samples", type=_optional_positive_int, help="Override data.max_samples for quick analysis.")
    parser.add_argument(
        "--num-workers",
        type=_positive_int,
        help="Number of CPU worker processes. Defaults to all detected CPUs.",
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=64,
        help="Records per multiprocessing task chunk.",
    )
    parser.add_argument(
        "--model-max-length",
        type=_positive_int,
        help="Override training.model_max_length for fit/truncation reporting.",
    )
    parser.add_argument(
        "--context-length",
        action="append",
        type=_positive_int,
        default=[],
        help="Additional context length to evaluate. May be repeated.",
    )
    parser.add_argument(
        "--respect-max-image-tokens",
        action="store_true",
        help=(
            "Apply data.max_image_tokens from the training YAML. By default the analyzer ignores it "
            "so raw dataset context-length statistics are computed before that filter."
        ),
    )
    parser.add_argument(
        "--image-open-timeout",
        type=_optional_positive_float,
        default=DEFAULT_IMAGE_OPEN_TIMEOUT_SECONDS,
        help=(
            "Seconds allowed for reading one image header before recording image_open_timeout. "
            "Use 0 to disable. Default: 30."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional matplotlib plots.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Plot file format when matplotlib is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis_from_yaml(
        args.training_yaml,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        model_max_length=args.model_max_length,
        context_lengths=args.context_length,
        respect_max_image_tokens=args.respect_max_image_tokens,
        image_open_timeout_seconds=args.image_open_timeout or None,
        write_plot_files=not args.no_plots,
        plot_format=args.plot_format,
    )
    counts = summary["counts"]
    output_files = summary["output_files"]
    print(f"records_loaded: {counts['records_loaded']}")
    print(f"usable_without_context_cap: {counts['usable_without_context_cap']}")
    print(f"usable_for_configured_training: {counts['usable_for_configured_training']}")
    print(f"records_needing_truncation_at_configured_length: {counts['records_needing_truncation_at_configured_length']}")
    print(f"summary_json: {output_files['summary_json']}")
    print(f"records_csv: {output_files['records_csv']}")
    print(f"broken_images_csv: {output_files['broken_images_csv']}")
    print(f"report_md: {output_files['report_md']}")
    if output_files["plots"]:
        print(f"plots: {len(output_files['plots'])}")


if __name__ == "__main__":
    main()
