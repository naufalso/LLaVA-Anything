"""Dataset utilities for LLaVa-Anything training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .processing_llava_anything import LlavaAnythingProcessor

IGNORE_INDEX = -100


def _process_rank() -> int:
    """Return the current distributed rank from torch or launcher environment."""

    if dist.is_initialized():
        return dist.get_rank()
    for env_name in ("RANK", "SLURM_PROCID"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            return int(env_value)
        except ValueError:
            continue
    return 0


def _is_main_process():
    return _process_rank() == 0


def _launcher_world_size() -> int:
    """Return the process count from torchrun/launcher environment when available."""

    for env_name in ("WORLD_SIZE", "SLURM_NTASKS"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            return max(1, int(env_value))
        except ValueError:
            continue
    return 1


def _load_json_records(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load JSON or JSONL training records from disk."""

    path = Path(path)
    if path.suffix == ".jsonl":
        print(f"Loading JSONL records from {path}")
        with path.open("r", encoding="utf-8") as f:
            records = []
            iterator = tqdm(
                f,
                desc=f"Loading {path}",
                unit=" records",
                disable=not _is_main_process(),
                mininterval=10.0 # Update the progress bar at most every 10 seconds to reduce overhead on large logs
            )
            for line_no, line in enumerate(iterator, start=1):
                line = line.strip()
                if not line:
                    continue
                if max_samples is not None and len(records) >= max_samples:
                    break
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    else:
        print(f"Loading JSON records from {path}")
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def _role_name(raw_role: str) -> str:
    """Normalize dataset role labels to chat-template role names."""

    role = raw_role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role


def _conversation_text(record: dict[str, Any]) -> tuple[str, str]:
    """Extract the first user prompt and assistant response from a record."""

    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Each record must contain a conversations list.")

    user_text: str | None = None
    assistant_text: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = _role_name(str(turn.get("from", turn.get("role", ""))))
        value = str(turn.get("value", turn.get("content", "")))
        if role == "user" and user_text is None:
            user_text = value
        elif role == "assistant" and assistant_text is None:
            assistant_text = value

    if user_text is None or assistant_text is None:
        raise ValueError("Each pretraining record must contain at least one user turn and one assistant turn.")
    return user_text, assistant_text


def _render_prefix(processor: LlavaAnythingProcessor, user_text: str, system_prompt: str | None = None) -> str:
    """Render the supervised-learning prompt prefix before assistant tokens."""

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_text})
    return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def _preview_record(
    processor: LlavaAnythingProcessor,
    record: dict[str, Any],
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Render a record into prompt prefix and target assistant text."""

    user_text, assistant_text = _conversation_text(record)
    return _render_prefix(processor, user_text, system_prompt), assistant_text


def log_preview_samples(dataset: "LlavaPretrainDataset", count: int = 2) -> None:
    """Print a small preview of templated training samples."""

    if count <= 0 or not _is_main_process():
        return
    limit = min(count, len(dataset))
    print(f"Previewing {limit} training sample(s) after prompt templating:")
    for index in range(limit):
        rendered_input, expected_output = _preview_record(
            dataset.processor,
            dataset.records[index],
            dataset.system_prompt,
        )
        print(f"Sample {index}")
        print("Rendered input:")
        print(rendered_input)
        print("Expected output:")
        print(expected_output)


_TOKENIZER_MODEL_MAX_LENGTH_SENTINEL = 10**20


def _resolve_model_max_length(processor: LlavaAnythingProcessor, model_max_length: Any | None = None) -> int | None:
    """Resolve the training max length from config or the tokenizer."""

    if model_max_length is not None:
        resolved = int(model_max_length)
        if resolved <= 0:
            raise ValueError("model_max_length must be a positive integer when provided.")
        return resolved

    tokenizer_max_length = getattr(processor.tokenizer, "model_max_length", None)
    if tokenizer_max_length is None:
        return None
    try:
        resolved = int(tokenizer_max_length)
    except (TypeError, ValueError, OverflowError):
        return None
    if resolved <= 0 or resolved >= _TOKENIZER_MODEL_MAX_LENGTH_SENTINEL:
        return None
    return resolved


def _tokenize_text(
    processor: LlavaAnythingProcessor,
    text: str,
    model_max_length: int | None = None,
) -> torch.LongTensor:
    """Tokenize text without adding tokenizer-level special tokens."""

    tokenizer_kwargs = {"add_special_tokens": False}
    if model_max_length is not None:
        tokenizer_kwargs.update({"truncation": True, "max_length": model_max_length})
    encoded = processor(text=text, return_tensors="pt", **tokenizer_kwargs)
    return encoded["input_ids"][0]


class LlavaPretrainDataset(Dataset):
    """Lazy reader for LLaVA-style image/conversation JSON records."""

    def __init__(
        self,
        data_path: str | Path,
        image_folder: str | Path,
        processor: LlavaAnythingProcessor,
        max_samples: int | None = None,
        available_images_only: bool = True,
        available_images_cache_dir: str | Path | None = None,
        available_images_num_workers: int = 0,
        refresh_available_images_cache: bool = False,
        require_image: bool = False,
        min_image_width: int | None = None,
        min_image_height: int | None = None,
        max_image_aspect_ratio: float | None = None,
        max_image_tokens: int | None = None,
        image_constraint_prefilter: bool = False,
        image_constraint_num_workers: int | None = None,
        image_token_mismatch_prefilter: bool = False,
        image_token_mismatch_num_workers: int | None = None,
        system_prompt: str | None = None,
        model_max_length: int | None = None,
    ) -> None:
        """Load record metadata and optionally filter examples with missing image files."""

        self.data_path = Path(data_path)
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.model_max_length = _resolve_model_max_length(processor, model_max_length)
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string when provided.")
        self.system_prompt = system_prompt or None
        self.max_samples = max_samples
        self.available_images_cache_dir = (
            Path(available_images_cache_dir) if available_images_cache_dir else None
        )
        self.available_images_num_workers = max(0, int(available_images_num_workers))
        self.refresh_available_images_cache = refresh_available_images_cache
        self.require_image = bool(require_image)
        self.min_image_width = int(min_image_width) if min_image_width is not None else None
        self.min_image_height = int(min_image_height) if min_image_height is not None else None
        self.max_image_aspect_ratio = (
            float(max_image_aspect_ratio) if max_image_aspect_ratio is not None else None
        )
        self.max_image_tokens = int(max_image_tokens) if max_image_tokens is not None else None
        self.image_constraint_prefilter = bool(image_constraint_prefilter)
        if image_constraint_num_workers is None:
            image_constraint_num_workers = self.available_images_num_workers
        self.image_constraint_num_workers = max(0, int(image_constraint_num_workers))
        self._image_constraint_warning_count = 0
        self._max_image_constraint_warnings = 20
        self.image_token_mismatch_prefilter = bool(image_token_mismatch_prefilter)
        if image_token_mismatch_num_workers is None:
            image_token_mismatch_num_workers = self.available_images_num_workers
        self.image_token_mismatch_num_workers = max(0, int(image_token_mismatch_num_workers))
        records = _load_json_records(self.data_path, max_samples)
        if self.require_image:
            original_count = len(records)
            records = [record for record in records if self._record_has_image(record)]
            skipped_text_count = original_count - len(records)
            if skipped_text_count:
                noun = "record" if skipped_text_count == 1 else "records"
                warnings.warn(
                    f"{skipped_text_count} text-only {noun} skipped before training because data.require_image is enabled.",
                    UserWarning,
                    stacklevel=2,
                )
        if available_images_only:
            records, skipped_count = self._filter_available_records(records)
            if skipped_count:
                noun = "image" if skipped_count == 1 else "images"
                warnings.warn(
                    f"{skipped_count} {noun} not found and skipping those before training.",
                    UserWarning,
                    stacklevel=2,
                )
        if max_samples is not None:
            records = records[:max_samples]
        if self.image_constraint_prefilter and self._has_image_constraints():
            records, image_constraint_skipped_count = self._filter_image_constraint_records(records)
            if _is_main_process():
                print(f"Image-constraint pre-filter skipped {image_constraint_skipped_count} sample(s) before training.")
            if image_constraint_skipped_count:
                noun = "sample" if image_constraint_skipped_count == 1 else "samples"
                warnings.warn(
                    f"{image_constraint_skipped_count} {noun} skipped before training because configured image constraints failed.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.image_token_mismatch_prefilter:
            records, image_token_skipped_count = self._filter_image_token_mismatch_records(records)
            if _is_main_process():
                print(f"Image-token pre-filter skipped {image_token_skipped_count} sample(s) before training.")
            if image_token_skipped_count:
                noun = "sample" if image_token_skipped_count == 1 else "samples"
                warnings.warn(
                    f"{image_token_skipped_count} {noun} skipped before training because image tokens or target labels would not fit.",
                    UserWarning,
                    stacklevel=2,
                )
        elif _is_main_process():
            print("Image-token pre-filter disabled; samples will be validated lazily during training.")
        self.records = records

    def __len__(self) -> int:
        """Return the number of available training records."""

        return len(self.records)

    def _record_image_path(self, record: dict[str, Any]) -> Path:
        """Resolve a record's image path relative to the dataset image folder."""

        image_name = record.get("image")
        if not image_name:
            raise ValueError("Pretraining records must include an image path.")
        return self.image_folder / str(image_name)

    def _record_has_image(self, record: dict[str, Any]) -> bool:
        """Return whether a record declares an image input."""

        return bool(record.get("image"))

    def _record_has_available_image(self, record: dict[str, Any]) -> bool:
        """Return whether a record points to an existing image file."""

        image_name = record.get("image")
        if not image_name:
            return False
        return (self.image_folder / str(image_name)).is_file()

    def _record_is_available(self, record: dict[str, Any]) -> bool:
        """Return whether a record should be kept after image availability filtering."""

        if not self._record_has_image(record):
            return True
        return self._record_has_available_image(record)

    def _available_images_cache_path(self) -> Path | None:
        """Return the cache file used for missing image indices."""

        if self.available_images_cache_dir is None:
            return None
        return self.available_images_cache_dir / "skipped_image_indices.json"

    def _image_token_mismatch_cache_path(self) -> Path | None:
        """Return the cache file used for image-token mismatch indices."""

        if self.available_images_cache_dir is None:
            return None
        return self.available_images_cache_dir / "skipped_image_token_indices.json"

    def _image_constraint_cache_path(self) -> Path | None:
        """Return the cache file used for image-constraint skipped indices."""

        if self.available_images_cache_dir is None:
            return None
        return self.available_images_cache_dir / "skipped_image_constraint_indices.json"

    def _available_images_cache_metadata(self, record_count: int) -> dict[str, Any]:
        """Build cache metadata that ties skipped indices to this dataset input."""

        try:
            data_stat = self.data_path.stat()
            data_size = data_stat.st_size
            data_mtime_ns = data_stat.st_mtime_ns
        except OSError:
            data_size = None
            data_mtime_ns = None

        return {
            "version": 1,
            "data_path": str(self.data_path.resolve()),
            "data_size": data_size,
            "data_mtime_ns": data_mtime_ns,
            "image_folder": str(self.image_folder.resolve()),
            "max_samples": self.max_samples,
            "require_image": self.require_image,
            "record_count": record_count,
        }

    def _image_token_mismatch_cache_metadata(self, record_count: int) -> dict[str, Any]:
        """Build cache metadata for image-token mismatch pre-filtering."""

        metadata = self._available_images_cache_metadata(record_count)
        metadata.update(
            {
                "cache_type": "image_token_mismatch",
                "filter_version": 2,
                "requires_supervised_targets": True,
                "image_token_mismatch_prefilter": self.image_token_mismatch_prefilter,
                "model_max_length": self.model_max_length,
                "system_prompt": self.system_prompt,
                "image_mode": getattr(self.processor, "image_mode", "fixed"),
                "image_grid_pinpoints": getattr(self.processor, "image_grid_pinpoints", None),
                "image_seq_length": getattr(self.processor, "image_seq_length", None),
                "patch_size": getattr(self.processor, "patch_size", None),
                "vision_feature_select_strategy": getattr(self.processor, "vision_feature_select_strategy", None),
                "num_additional_image_tokens": getattr(self.processor, "num_additional_image_tokens", None),
            }
        )
        return metadata

    def _image_constraint_cache_metadata(self, record_count: int) -> dict[str, Any]:
        """Build cache metadata for image-constraint pre-filtering."""

        metadata = self._available_images_cache_metadata(record_count)
        metadata.update(
            {
                "cache_type": "image_constraint",
                "filter_version": 1,
                "image_constraint_prefilter": self.image_constraint_prefilter,
                "min_image_width": self.min_image_width,
                "min_image_height": self.min_image_height,
                "max_image_aspect_ratio": self.max_image_aspect_ratio,
                "max_image_tokens": self.max_image_tokens,
                "image_mode": getattr(self.processor, "image_mode", "fixed"),
                "image_grid_pinpoints": getattr(self.processor, "image_grid_pinpoints", None),
                "image_seq_length": getattr(self.processor, "image_seq_length", None),
                "patch_size": getattr(self.processor, "patch_size", None),
                "vision_feature_select_strategy": getattr(self.processor, "vision_feature_select_strategy", None),
                "num_additional_image_tokens": getattr(self.processor, "num_additional_image_tokens", None),
            }
        )
        return metadata

    def _load_skipped_indices_cache(self, records: list[dict[str, Any]]) -> set[int] | None:
        """Load cached missing-image indices when the cache still matches the dataset."""

        cache_path = self._available_images_cache_path()
        if (
            cache_path is None
            or self.refresh_available_images_cache
            or not cache_path.is_file()
        ):
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"Ignoring unavailable image cache at {cache_path}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return None

        expected_metadata = self._available_images_cache_metadata(len(records))
        if cache.get("metadata") != expected_metadata:
            return None

        skipped_indices = cache.get("skipped_indices")
        if not isinstance(skipped_indices, list) or not all(
            isinstance(index, int) for index in skipped_indices
        ):
            warnings.warn(
                f"Ignoring malformed unavailable image cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        skipped = set(skipped_indices)
        if any(index < 0 or index >= len(records) for index in skipped):
            warnings.warn(
                f"Ignoring out-of-range unavailable image cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        if _is_main_process():
            print(
                f"Loaded {len(skipped)} cached skipped image index/indices from {cache_path}"
            )
        return skipped

    def _wait_for_skipped_indices_cache(
        self,
        records: list[dict[str, Any]],
        poll_seconds: float = 10.0,
    ) -> set[int]:
        """Wait for rank 0 to write a valid skipped-index cache."""

        cache_path = self._available_images_cache_path()
        if cache_path is None:
            raise RuntimeError("Cannot wait for skipped image cache without a cache path.")

        while True:
            if cache_path.is_file():
                cached_skipped_indices = self._load_skipped_indices_cache(records)
                if cached_skipped_indices is not None:
                    return cached_skipped_indices
            time.sleep(poll_seconds)

    def _save_skipped_indices_cache(
        self,
        records: list[dict[str, Any]],
        skipped_indices: list[int],
    ) -> None:
        """Persist missing-image indices for faster restarts on the same dataset."""

        cache_path = self._available_images_cache_path()
        if cache_path is None or not _is_main_process():
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "metadata": self._available_images_cache_metadata(len(records)),
            "skipped_indices": skipped_indices,
        }
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if _is_main_process():
            print(
                f"Saved {len(skipped_indices)} skipped image index/indices to {cache_path}"
            )

    def _load_image_token_mismatch_cache(self, records: list[dict[str, Any]]) -> set[int] | None:
        """Load cached image-token mismatch indices when the cache still matches."""

        cache_path = self._image_token_mismatch_cache_path()
        if (
            cache_path is None
            or self.refresh_available_images_cache
            or not cache_path.is_file()
        ):
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"Ignoring image-token mismatch cache at {cache_path}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return None

        expected_metadata = self._image_token_mismatch_cache_metadata(len(records))
        if cache.get("metadata") != expected_metadata:
            return None

        skipped_indices = cache.get("skipped_indices")
        if not isinstance(skipped_indices, list) or not all(
            isinstance(index, int) for index in skipped_indices
        ):
            warnings.warn(
                f"Ignoring malformed image-token mismatch cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        skipped = set(skipped_indices)
        if any(index < 0 or index >= len(records) for index in skipped):
            warnings.warn(
                f"Ignoring out-of-range image-token mismatch cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        if _is_main_process():
            print(f"Loaded {len(skipped)} cached image-token mismatch index/indices from {cache_path}")
        return skipped

    def _wait_for_image_token_mismatch_cache(
        self,
        records: list[dict[str, Any]],
        poll_seconds: float = 10.0,
    ) -> set[int]:
        """Wait for rank 0 to write a valid image-token mismatch cache."""

        cache_path = self._image_token_mismatch_cache_path()
        if cache_path is None:
            raise RuntimeError("Cannot wait for image-token mismatch cache without a cache path.")

        while True:
            if cache_path.is_file():
                cached_skipped_indices = self._load_image_token_mismatch_cache(records)
                if cached_skipped_indices is not None:
                    return cached_skipped_indices
            time.sleep(poll_seconds)

    def _save_image_token_mismatch_cache(
        self,
        records: list[dict[str, Any]],
        skipped_indices: list[int],
    ) -> None:
        """Persist image-token mismatch indices for faster distributed restarts."""

        cache_path = self._image_token_mismatch_cache_path()
        if cache_path is None or not _is_main_process():
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "metadata": self._image_token_mismatch_cache_metadata(len(records)),
            "skipped_indices": skipped_indices,
        }
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if _is_main_process():
            print(f"Saved {len(skipped_indices)} image-token mismatch index/indices to {cache_path}")

    def _load_image_constraint_cache(self, records: list[dict[str, Any]]) -> set[int] | None:
        """Load cached image-constraint indices when the cache still matches."""

        cache_path = self._image_constraint_cache_path()
        if (
            cache_path is None
            or self.refresh_available_images_cache
            or not cache_path.is_file()
        ):
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"Ignoring image-constraint cache at {cache_path}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return None

        expected_metadata = self._image_constraint_cache_metadata(len(records))
        if cache.get("metadata") != expected_metadata:
            return None

        skipped_indices = cache.get("skipped_indices")
        if not isinstance(skipped_indices, list) or not all(
            isinstance(index, int) for index in skipped_indices
        ):
            warnings.warn(
                f"Ignoring malformed image-constraint cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        skipped = set(skipped_indices)
        if any(index < 0 or index >= len(records) for index in skipped):
            warnings.warn(
                f"Ignoring out-of-range image-constraint cache at {cache_path}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        if _is_main_process():
            print(f"Loaded {len(skipped)} cached image-constraint index/indices from {cache_path}")
        return skipped

    def _wait_for_image_constraint_cache(
        self,
        records: list[dict[str, Any]],
        poll_seconds: float = 10.0,
    ) -> set[int]:
        """Wait for rank 0 to write a valid image-constraint cache."""

        cache_path = self._image_constraint_cache_path()
        if cache_path is None:
            raise RuntimeError("Cannot wait for image-constraint cache without a cache path.")

        while True:
            if cache_path.is_file():
                cached_skipped_indices = self._load_image_constraint_cache(records)
                if cached_skipped_indices is not None:
                    return cached_skipped_indices
            time.sleep(poll_seconds)

    def _save_image_constraint_cache(
        self,
        records: list[dict[str, Any]],
        skipped_indices: list[int],
    ) -> None:
        """Persist image-constraint indices for faster distributed restarts."""

        cache_path = self._image_constraint_cache_path()
        if cache_path is None or not _is_main_process():
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "metadata": self._image_constraint_cache_metadata(len(records)),
            "skipped_indices": skipped_indices,
        }
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if _is_main_process():
            print(f"Saved {len(skipped_indices)} image-constraint index/indices to {cache_path}")

    def _filter_available_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Keep text-only records and image records with existing files."""

        cached_skipped_indices = self._load_skipped_indices_cache(records)
        if cached_skipped_indices is not None:
            available = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return available, len(cached_skipped_indices)

        if (
            not _is_main_process()
            and _launcher_world_size() > 1
            and self._available_images_cache_path() is not None
            and not self.refresh_available_images_cache
        ):
            cached_skipped_indices = self._wait_for_skipped_indices_cache(records)
            available = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return available, len(cached_skipped_indices)

        available: list[dict[str, Any]] = []
        skipped_indices: list[int] = []
        worker_count = self.available_images_num_workers
        if worker_count > 1 and _is_main_process():
            print(f"Checking image availability with {worker_count} workers")

        def collect(is_available_by_index) -> None:
            for index, (record, is_available) in enumerate(
                zip(records, is_available_by_index)
            ):
                if is_available:
                    available.append(record)
                else:
                    skipped_indices.append(index)

        if worker_count <= 1:
            is_available_iter = (
                self._record_is_available(record)
                for record in tqdm(
                    records,
                    disable=not _is_main_process(),
                    mininterval=10.0,
                    desc="Checking for available images",
                )
            )
            collect(is_available_iter)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                is_available_iter = tqdm(
                    executor.map(self._record_is_available, records),
                    total=len(records),
                    disable=not _is_main_process(),
                    mininterval=10.0,
                    desc="Checking for available images",
                )
                collect(is_available_iter)
        self._save_skipped_indices_cache(records, skipped_indices)
        return available, len(skipped_indices)

    def _record_image_size(self, record: dict[str, Any]) -> list[int]:
        """Read a record image's original [height, width] without full preprocessing."""

        image_path = self._record_image_path(record)
        with Image.open(image_path) as image:
            width, height = image.size
        return [height, width]

    def _load_image(self, record: dict[str, Any]) -> Image.Image:
        """Load a record image as RGB PIL data."""

        image_path = self._record_image_path(record)
        return Image.open(image_path).convert("RGB")

    def _has_image_constraints(self) -> bool:
        """Return whether any image-quality constraint is configured."""

        return any(
            value is not None
            for value in (
                self.min_image_width,
                self.min_image_height,
                self.max_image_aspect_ratio,
                self.max_image_tokens,
            )
        )

    def _tokenizer_kwargs(self, truncate: bool = True) -> dict[str, Any]:
        """Return text-tokenization kwargs used by all training samples."""

        kwargs: dict[str, Any] = {"add_special_tokens": False}
        if truncate and self.model_max_length is not None:
            kwargs.update({"truncation": True, "max_length": self.model_max_length})
        return kwargs

    def _tokenize_expanded_image_text(
        self,
        text: str,
        image_size: list[int] | None = None,
        truncate: bool = True,
    ) -> torch.LongTensor:
        """Tokenize text after expanding image markers without preprocessing pixels."""

        image_sizes = [image_size] if image_size is not None else None
        expanded_text = self.processor._expand_image_tokens(text, image_sizes)
        encoded = self.processor.tokenizer(
            expanded_text,
            return_tensors="pt",
            **self._tokenizer_kwargs(truncate=truncate),
        )
        return encoded["input_ids"][0]

    def _build_labels(self, input_ids: torch.LongTensor, original_positions: torch.LongTensor) -> torch.LongTensor:
        """Mask prompt and image-token positions, leaving assistant tokens supervised."""

        labels = input_ids.clone()
        labels[original_positions < 0] = IGNORE_INDEX
        labels[input_ids == self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)] = IGNORE_INDEX
        return labels

    def _truncate_for_supervised_training(
        self,
        input_ids: torch.LongTensor,
        prefix_length: int,
        max_length: int,
    ) -> tuple[torch.LongTensor, torch.LongTensor] | None:
        """Truncate while preserving image tokens and at least one assistant target token."""

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        image_positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten().tolist()
        image_position_set = set(image_positions)
        if len(image_positions) >= max_length:
            return None

        target_positions = [
            index
            for index in range(prefix_length, input_ids.shape[0])
            if index not in image_position_set
        ]
        if not target_positions:
            return None

        text_budget = max_length - len(image_positions)
        target_budget = min(len(target_positions), text_budget)
        if target_budget <= 0:
            return None

        if getattr(self.processor.tokenizer, "truncation_side", "right") == "left":
            kept_target_positions = target_positions[-target_budget:]
        else:
            kept_target_positions = target_positions[:target_budget]

        prompt_budget = text_budget - len(kept_target_positions)
        prompt_positions = [
            index
            for index in range(prefix_length)
            if index not in image_position_set
        ]
        if getattr(self.processor.tokenizer, "truncation_side", "right") == "left":
            kept_prompt_positions = prompt_positions[-prompt_budget:] if prompt_budget else []
        else:
            kept_prompt_positions = prompt_positions[:prompt_budget]

        selected_positions = sorted([*image_positions, *kept_prompt_positions, *kept_target_positions])
        selected = torch.tensor(selected_positions, dtype=torch.long, device=input_ids.device)
        truncated_input_ids = input_ids.index_select(0, selected)
        original_positions = selected.clone()
        original_positions[original_positions < prefix_length] = -1
        labels = self._build_labels(truncated_input_ids, original_positions)
        if not torch.any(labels != IGNORE_INDEX):
            return None
        return truncated_input_ids, labels

    def _record_has_matching_image_tokens(self, record: dict[str, Any]) -> bool:
        """Return whether a record keeps matching image tokens and target labels."""

        prefix, assistant_text = _preview_record(self.processor, record, self.system_prompt)
        eos = self.processor.tokenizer.eos_token or ""
        full_text = f"{prefix}{assistant_text}{eos}"
        has_image = self._record_has_image(record)
        image_token = self.processor.image_token
        if not has_image:
            if image_token in full_text:
                return False
            input_ids = _tokenize_text(self.processor, full_text, None)
            prefix_length = int(_tokenize_text(self.processor, prefix, None).shape[0])
        else:
            image_size = self._record_image_size(record) if getattr(self.processor, "image_mode", "fixed") == "anyres" else None
            input_ids = self._tokenize_expanded_image_text(full_text, image_size, truncate=False)
            prefix_length = int(self._tokenize_expanded_image_text(prefix, image_size, truncate=False).shape[0])

        labels: torch.LongTensor | None = None
        if self.model_max_length is not None and input_ids.shape[0] > self.model_max_length:
            truncated = self._truncate_for_supervised_training(
                input_ids,
                prefix_length,
                self.model_max_length,
            )
            if truncated is None:
                return False
            input_ids, labels = truncated
        if labels is None:
            original_positions = torch.arange(input_ids.shape[0], dtype=torch.long)
            original_positions[original_positions < prefix_length] = -1
            labels = self._build_labels(input_ids, original_positions)
        if not torch.any(labels != IGNORE_INDEX):
            return False

        expected = self.processor._num_image_tokens(image_size) if has_image else 0
        actual = self._image_token_count(input_ids)
        return actual == expected

    def _filter_image_token_mismatch_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Pre-filter records whose image-token span would be truncated."""

        if self.model_max_length is None:
            return records, 0

        cached_skipped_indices = self._load_image_token_mismatch_cache(records)
        if cached_skipped_indices is not None:
            filtered = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return filtered, len(cached_skipped_indices)

        if (
            not _is_main_process()
            and _launcher_world_size() > 1
            and self._image_token_mismatch_cache_path() is not None
            and not self.refresh_available_images_cache
        ):
            cached_skipped_indices = self._wait_for_image_token_mismatch_cache(records)
            filtered = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return filtered, len(cached_skipped_indices)

        filtered: list[dict[str, Any]] = []
        skipped_indices: list[int] = []
        worker_count = self.image_token_mismatch_num_workers
        if worker_count > 1 and _is_main_process():
            print(f"Checking image-token lengths with {worker_count} workers")

        def is_record_valid(record: dict[str, Any]) -> bool:
            try:
                return self._record_has_matching_image_tokens(record)
            except Exception as exc:
                record_id = record.get("id", "<unknown>")
                warnings.warn(
                    f"Skipping record during image-token pre-filter: record_id={record_id!r}, error={exc}",
                    UserWarning,
                    stacklevel=2,
                )
                return False

        if worker_count <= 1:
            keep_iter = (
                is_record_valid(record)
                for record in tqdm(
                    records,
                    disable=not _is_main_process(),
                    mininterval=10.0,
                    desc="Checking image-token lengths",
                )
            )
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            keep_iter = tqdm(
                executor.map(is_record_valid, records),
                total=len(records),
                disable=not _is_main_process(),
                mininterval=10.0,
                desc="Checking image-token lengths",
            )

        try:
            for index, (record, keep) in enumerate(zip(records, keep_iter)):
                if keep:
                    filtered.append(record)
                else:
                    skipped_indices.append(index)
        finally:
            if worker_count > 1:
                executor.shutdown(wait=True)
        self._save_image_token_mismatch_cache(records, skipped_indices)
        return filtered, len(skipped_indices)

    def _expected_image_token_count(self, image_inputs: dict[str, torch.Tensor]) -> int:
        """Return how many expanded image tokens are needed for processed images."""

        if "image_sizes" in image_inputs:
            image_sizes = image_inputs["image_sizes"]
            if image_sizes.dim() == 1:
                image_sizes = image_sizes.unsqueeze(0)
            return sum(
                self.processor._num_image_tokens(image_size.tolist())
                for image_size in image_sizes
            )
        if "pixel_values" in image_inputs:
            return self.processor._num_image_tokens()
        return 0

    def _image_token_count(self, input_ids: torch.LongTensor) -> int:
        """Count expanded image tokens in tokenized text."""

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return int((input_ids == image_token_id).sum().item())

    def _warn_image_constraint_skip(self, record: dict[str, Any], reason: str) -> None:
        """Warn about skipped image-quality constraints without flooding logs."""

        if self._image_constraint_warning_count >= self._max_image_constraint_warnings:
            return
        self._image_constraint_warning_count += 1
        record_id = record.get("id", "<unknown>")
        image_path = self._record_image_path(record)
        suffix = ""
        if self._image_constraint_warning_count == self._max_image_constraint_warnings:
            suffix = " Further image-constraint skip warnings will be suppressed."
        warnings.warn(
            f"Skipping record because image constraints failed: record_id={record_id!r}, "
            f"image_path={str(image_path)!r}, reason={reason}.{suffix}",
            UserWarning,
            stacklevel=2,
        )

    def _image_constraint_failure_reason(self, height: int, width: int) -> str | None:
        """Return the first configured image-constraint failure, if any."""

        if self.min_image_width is not None and width < self.min_image_width:
            return f"width {width} < min_image_width {self.min_image_width}"
        if self.min_image_height is not None and height < self.min_image_height:
            return f"height {height} < min_image_height {self.min_image_height}"
        if self.max_image_aspect_ratio is not None:
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > self.max_image_aspect_ratio:
                return f"aspect_ratio {aspect_ratio:.3g} > max_image_aspect_ratio {self.max_image_aspect_ratio}"
        if self.max_image_tokens is not None:
            image_tokens = self.processor._num_image_tokens([height, width])
            if image_tokens > self.max_image_tokens:
                return f"image_tokens {image_tokens} > max_image_tokens {self.max_image_tokens}"
        return None

    def _record_image_constraint_failure_reason(self, record: dict[str, Any]) -> str | None:
        """Return why a record fails image constraints, or None when it passes."""

        if not self._record_has_image(record):
            return None
        try:
            height, width = self._record_image_size(record)
        except Exception as exc:
            return f"image_open_error: {exc}"
        return self._image_constraint_failure_reason(height, width)

    def _filter_image_constraint_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Pre-filter records whose images violate configured stability constraints."""

        cached_skipped_indices = self._load_image_constraint_cache(records)
        if cached_skipped_indices is not None:
            filtered = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return filtered, len(cached_skipped_indices)

        if (
            not _is_main_process()
            and _launcher_world_size() > 1
            and self._image_constraint_cache_path() is not None
            and not self.refresh_available_images_cache
        ):
            cached_skipped_indices = self._wait_for_image_constraint_cache(records)
            filtered = [
                record
                for index, record in enumerate(records)
                if index not in cached_skipped_indices
            ]
            return filtered, len(cached_skipped_indices)

        filtered: list[dict[str, Any]] = []
        skipped_indices: list[int] = []
        worker_count = self.image_constraint_num_workers
        if worker_count > 1 and _is_main_process():
            print(f"Checking image constraints with {worker_count} workers")

        def keep_record(record: dict[str, Any]) -> bool:
            return self._record_image_constraint_failure_reason(record) is None

        def collect(keep_by_index) -> None:
            for index, (record, keep) in enumerate(zip(records, keep_by_index)):
                if keep:
                    filtered.append(record)
                else:
                    skipped_indices.append(index)

        if worker_count <= 1:
            keep_iter = (
                keep_record(record)
                for record in tqdm(
                    records,
                    disable=not _is_main_process(),
                    mininterval=10.0,
                    desc="Checking image constraints",
                )
            )
            collect(keep_iter)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                keep_iter = tqdm(
                    executor.map(keep_record, records),
                    total=len(records),
                    disable=not _is_main_process(),
                    mininterval=10.0,
                    desc="Checking image constraints",
                )
                collect(keep_iter)

        self._save_image_constraint_cache(records, skipped_indices)
        return filtered, len(skipped_indices)

    def _record_satisfies_image_constraints(
        self,
        record: dict[str, Any],
        image: Image.Image,
    ) -> bool:
        """Return whether an image is within configured training-stability constraints."""

        width, height = image.size
        reason = self._image_constraint_failure_reason(height, width)
        if reason is not None:
            self._warn_image_constraint_skip(record, reason)
            return False
        return True

    def _warn_image_token_mismatch(
        self,
        record: dict[str, Any],
        actual: int,
        expected: int,
        action: str,
    ) -> None:
        """Warn with enough record context to debug image-token mismatches."""

        record_id = record.get("id", "<unknown>")
        image_path = self._record_image_path(record)
        warnings.warn(
            "Image token expansion was truncated or mismatched before model forward: "
            f"record_id={record_id!r}, image_path={str(image_path)!r}, "
            f"tokens={actual}, expected_features={expected}, model_max_length={self.model_max_length}. "
            f"{action}",
            UserWarning,
            stacklevel=2,
        )

    def _warn_text_only_image_token_skip(self, record: dict[str, Any]) -> None:
        """Warn when a text-only record contains the multimodal placeholder."""

        record_id = record.get("id", "<unknown>")
        warnings.warn(
            f"Skipping text-only record because it contains the image token: record_id={record_id!r}.",
            UserWarning,
            stacklevel=2,
        )

    def _sample_metadata(self, index: int, record: dict[str, Any]) -> dict[str, Any]:
        """Return compact source metadata for diagnostics outside model forward."""

        metadata: dict[str, Any] = {"record_index": index}
        if "id" in record:
            metadata["record_id"] = str(record["id"])
        if self._record_has_image(record):
            metadata["image_path"] = str(self._record_image_path(record))
        return metadata

    def _build_sample(self, index: int) -> dict[str, torch.Tensor | Any] | None:
        """Build one supervised training sample, returning None when it must be skipped."""

        record = self.records[index]
        prefix, assistant_text = _preview_record(self.processor, record, self.system_prompt)
        eos = self.processor.tokenizer.eos_token or ""
        full_text = f"{prefix}{assistant_text}{eos}"

        has_image = self._record_has_image(record)
        image_token = self.processor.image_token
        if not has_image and image_token in full_text:
            self._warn_text_only_image_token_skip(record)
            return None

        image = self._load_image(record) if has_image else None
        if image is not None and not self._record_satisfies_image_constraints(record, image):
            return None
        labels = None
        if has_image and getattr(self.processor, "image_mode", "fixed") == "anyres":
            full_inputs = self.processor(images=image, text=full_text, return_tensors="pt", **self._tokenizer_kwargs(truncate=False))
            prefix_inputs = self.processor(images=image, text=prefix, return_tensors="pt", **self._tokenizer_kwargs(truncate=False))
            input_ids = full_inputs["input_ids"][0]
            prefix_length = int(prefix_inputs["input_ids"][0].shape[0])
            image_inputs = full_inputs
        else:
            input_ids = _tokenize_text(self.processor, full_text, None)
            prefix_length = int(_tokenize_text(self.processor, prefix, None).shape[0])
            image_inputs = self.processor(images=image, return_tensors="pt") if image is not None else {}

        if self.model_max_length is not None and input_ids.shape[0] > self.model_max_length:
            truncated = self._truncate_for_supervised_training(
                input_ids,
                prefix_length,
                self.model_max_length,
            )
            if truncated is None:
                record_id = record.get("id", "<unknown>")
                warnings.warn(
                    f"Skipping record because no assistant target tokens fit within model_max_length: record_id={record_id!r}.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            input_ids, labels = truncated

        if has_image:
            expected = self._expected_image_token_count(image_inputs)
            actual = self._image_token_count(input_ids)
            if actual != expected:
                self._warn_image_token_mismatch(
                    record,
                    actual,
                    expected,
                    "Skipping this sample to avoid training on a semantically altered example.",
                )
                return None

        if labels is None:
            original_positions = torch.arange(input_ids.shape[0], dtype=torch.long)
            original_positions[original_positions < prefix_length] = -1
            labels = self._build_labels(input_ids, original_positions)
        if not torch.any(labels != IGNORE_INDEX):
            record_id = record.get("id", "<unknown>")
            warnings.warn(
                f"Skipping record because it has no assistant target tokens after masking: record_id={record_id!r}.",
                UserWarning,
                stacklevel=2,
            )
            return None

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "_metadata": self._sample_metadata(index, record),
        }
        if image_inputs:
            sample["pixel_values"] = image_inputs["pixel_values"][0]
        if "image_sizes" in image_inputs:
            sample["image_sizes"] = image_inputs["image_sizes"][0]
        return sample

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | Any]:
        """Build one supervised training sample, skipping unrecoverable records."""

        for offset in range(len(self.records)):
            sample = self._build_sample((index + offset) % len(self.records))
            if sample is not None:
                return sample
        raise ValueError("No usable training records remain after image-token validation.")


@dataclass
class LlavaPretrainDataCollator:
    tokenizer: Any
    include_metadata: bool = False

    def _pad_sequence(self, tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Pad token sequences while respecting the tokenizer padding side."""

        if getattr(self.tokenizer, "padding_side", "right") == "left":
            tensors = [torch.flip(tensor, dims=[0]) for tensor in tensors]
            padded = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
            return torch.flip(padded, dims=[1])
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def __call__(
        self,
        instances: list[dict[str, torch.Tensor | Any] | None],
    ) -> dict[str, torch.Tensor | Any]:
        """Collate text plus any available image tensors into a batch."""

        instances = [instance for instance in instances if instance is not None]
        if not instances:
            raise ValueError("All samples in this batch were skipped before collation.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids = self._pad_sequence([instance["input_ids"] for instance in instances], pad_token_id)
        labels = self._pad_sequence([instance["labels"] for instance in instances], IGNORE_INDEX)
        if not torch.any(labels != IGNORE_INDEX):
            raise ValueError("Batch has no supervised target tokens after masking.")
        attention_mask = input_ids.ne(pad_token_id)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if self.include_metadata:
            metadata = [instance.get("_metadata") for instance in instances]
            if any(item is not None for item in metadata):
                batch["_metadata"] = metadata
        image_instances = [instance for instance in instances if "pixel_values" in instance]
        if image_instances:
            pixel_value_tensors = [instance["pixel_values"] for instance in image_instances]
            if pixel_value_tensors[0].dim() == 4:
                max_patches = max(tensor.shape[0] for tensor in pixel_value_tensors)
                padded_pixel_values = []
                for tensor in pixel_value_tensors:
                    if tensor.shape[0] < max_patches:
                        padding = torch.zeros(
                            (max_patches - tensor.shape[0], *tensor.shape[1:]),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        tensor = torch.cat([tensor, padding], dim=0)
                    padded_pixel_values.append(tensor)
                batch["pixel_values"] = torch.stack(padded_pixel_values)
            else:
                batch["pixel_values"] = torch.stack(pixel_value_tensors)

        image_size_tensors = [instance["image_sizes"] for instance in image_instances if "image_sizes" in instance]
        if image_size_tensors:
            batch["image_sizes"] = torch.stack(image_size_tensors)
        return batch


__all__ = [
    "IGNORE_INDEX",
    "LlavaPretrainDataCollator",
    "LlavaPretrainDataset",
    "log_preview_samples",
]
