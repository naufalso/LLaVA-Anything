"""Training utilities for LLaVa-Anything."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .builder import config_from_yaml_dict, load_yaml, model_from_yaml_dict, processor_from_yaml_dict
from .modeling_llava_anything import LlavaAnythingForConditionalGeneration
from .processing_llava_anything import LlavaAnythingProcessor

import torch.distributed as dist
from tqdm import tqdm

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

    if count <= 0:
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
        refresh_available_images_cache: bool = False,
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
        self.refresh_available_images_cache = refresh_available_images_cache
        records = _load_json_records(self.data_path, max_samples)
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

    def _available_images_cache_path(self) -> Path | None:
        """Return the cache file used for missing image indices."""

        if self.available_images_cache_dir is None:
            return None
        return self.available_images_cache_dir / "skipped_image_indices.json"

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
            "record_count": record_count,
        }

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

        available: list[dict[str, Any]] = []
        skipped_indices: list[int] = []
        for index, record in enumerate(
            tqdm(
                records,
                disable=not _is_main_process(),
                mininterval=10.0,
                desc="Checking for available images",
            )
        ):
            if not self._record_has_image(record):
                available.append(record)
            elif self._record_has_available_image(record):
                available.append(record)
            else:
                skipped_indices.append(index)
        self._save_skipped_indices_cache(records, skipped_indices)
        return available, len(skipped_indices)

    def _load_image(self, record: dict[str, Any]) -> Image.Image:
        """Load a record image as RGB PIL data."""

        image_path = self._record_image_path(record)
        return Image.open(image_path).convert("RGB")

    def _tokenizer_kwargs(self) -> dict[str, Any]:
        """Return text-tokenization kwargs used by all training samples."""

        kwargs: dict[str, Any] = {"add_special_tokens": False}
        if self.model_max_length is not None:
            kwargs.update({"truncation": True, "max_length": self.model_max_length})
        return kwargs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one supervised training sample, with images when present."""

        record = self.records[index]
        prefix, assistant_text = _preview_record(self.processor, record, self.system_prompt)
        eos = self.processor.tokenizer.eos_token or ""
        full_text = f"{prefix}{assistant_text}{eos}"

        has_image = self._record_has_image(record)
        image_token = self.processor.image_token
        if not has_image and image_token in full_text:
            raise ValueError("Text-only records must not contain the image token.")

        if has_image and getattr(self.processor, "image_mode", "fixed") == "anyres":
            image = self._load_image(record)
            full_inputs = self.processor(images=image, text=full_text, return_tensors="pt", **self._tokenizer_kwargs())
            prefix_inputs = self.processor(images=image, text=prefix, return_tensors="pt", **self._tokenizer_kwargs())
            input_ids = full_inputs["input_ids"][0]
            prefix_ids = prefix_inputs["input_ids"][0]
            image_inputs = full_inputs
        else:
            input_ids = _tokenize_text(self.processor, full_text, self.model_max_length)
            prefix_ids = _tokenize_text(self.processor, prefix, self.model_max_length)
            image_inputs = self.processor(images=self._load_image(record), return_tensors="pt") if has_image else {}

        labels = input_ids.clone()
        labels[: prefix_ids.shape[0]] = IGNORE_INDEX
        labels[input_ids == self.processor.tokenizer.convert_tokens_to_ids(image_token)] = IGNORE_INDEX

        sample = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if image_inputs:
            sample["pixel_values"] = image_inputs["pixel_values"][0]
        if "image_sizes" in image_inputs:
            sample["image_sizes"] = image_inputs["image_sizes"][0]
        return sample


@dataclass
class LlavaPretrainDataCollator:
    tokenizer: Any

    def _pad_sequence(self, tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Pad token sequences while respecting the tokenizer padding side."""

        if getattr(self.tokenizer, "padding_side", "right") == "left":
            tensors = [torch.flip(tensor, dims=[0]) for tensor in tensors]
            padded = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
            return torch.flip(padded, dims=[1])
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate text plus any available image tensors into a batch."""

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids = self._pad_sequence([instance["input_ids"] for instance in instances], pad_token_id)
        labels = self._pad_sequence([instance["labels"] for instance in instances], IGNORE_INDEX)
        attention_mask = input_ids.ne(pad_token_id)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
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


def apply_trainable_modules(model: LlavaAnythingForConditionalGeneration, trainable_modules: str = "projector") -> list[str]:
    """Freeze all parameters except the requested trainable module groups."""

    modules = {part.strip() for part in trainable_modules.split(",") if part.strip()}
    supported = {"projector", "vision_tower", "language_model", "full"}
    unknown = modules - supported
    if unknown:
        raise ValueError(f"Unsupported trainable module selection: {sorted(unknown)}")

    model.requires_grad_(False)
    if "full" in modules:
        model.requires_grad_(True)
    if "projector" in modules or "full" in modules:
        model.multi_modal_projector.requires_grad_(True)
    if "vision_tower" in modules or "full" in modules:
        model.vision_tower.requires_grad_(True)
    if "language_model" in modules or "full" in modules:
        model.language_model.requires_grad_(True)

    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


@dataclass
class LlavaPretrainingResult:
    train_result: Any
    output_dir: Path
    trainable_parameter_names: list[str]


def configure_wandb(training_section: dict[str, Any], wandb_section: dict[str, Any] | None) -> None:
    """Apply optional Weights & Biases settings to training args and environment variables."""

    if wandb_section is None:
        return

    wandb_section = dict(wandb_section)
    if wandb_section.get("enabled") is False:
        training_section.setdefault("report_to", [])
        return

    training_section.setdefault("report_to", ["wandb"])
    if isinstance(training_section["report_to"], str):
        training_section["report_to"] = [training_section["report_to"]]
    if "wandb" not in training_section["report_to"]:
        training_section["report_to"] = [*training_section["report_to"], "wandb"]

    env_map = {
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "mode": "WANDB_MODE",
        "name": "WANDB_NAME",
    }
    for key, env_name in env_map.items():
        value = wandb_section.get(key)
        if value is not None:
            os.environ[env_name] = str(value)
    if wandb_section.get("name") and "run_name" not in training_section:
        training_section["run_name"] = str(wandb_section["name"])


def _coerce_training_arguments(training_section: dict[str, Any]) -> TrainingArguments:
    """Convert a YAML training section into Hugging Face TrainingArguments."""

    kwargs = dict(training_section)
    kwargs.setdefault("remove_unused_columns", False)
    kwargs.setdefault("report_to", [])
    kwargs.setdefault("save_strategy", "no")
    if kwargs.get("save_strategy") is False:
        kwargs["save_strategy"] = "no"
    kwargs.setdefault("logging_steps", 1)
    kwargs.setdefault("disable_tqdm", True)
    if "output_dir" not in kwargs:
        raise ValueError("training.output_dir is required.")
    return TrainingArguments(**kwargs)


def _resolve_resume_from_checkpoint(training_args: TrainingArguments) -> str | bool | None:
    """Resolve the checkpoint to resume from, auto-detecting the latest checkpoint by default."""

    explicit_resume = getattr(training_args, "resume_from_checkpoint", None)
    if explicit_resume is not None:
        return explicit_resume

    output_dir = Path(training_args.output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def _load_training_yaml(path: str | Path) -> dict[str, Any]:
    """Load a training YAML file and require a top-level mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _coerce_torch_dtype(value: Any) -> Any:
    """Convert string dtype names into torch dtype objects when possible."""

    if not isinstance(value, str) or value == "auto":
        return value
    if hasattr(torch, value):
        dtype = getattr(torch, value)
        if isinstance(dtype, torch.dtype):
            return dtype
    return value


def _coerce_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recursively coerce model-loading dtype kwargs from YAML-friendly strings."""

    if model_kwargs is None:
        return None
    coerced = dict(model_kwargs)
    if "torch_dtype" in coerced:
        coerced["torch_dtype"] = _coerce_torch_dtype(coerced["torch_dtype"])
    for nested_key in ("text_model_kwargs", "vision_model_kwargs"):
        if isinstance(coerced.get(nested_key), dict) and "torch_dtype" in coerced[nested_key]:
            coerced[nested_key] = dict(coerced[nested_key])
            coerced[nested_key]["torch_dtype"] = _coerce_torch_dtype(coerced[nested_key]["torch_dtype"])
    return coerced


def _build_model_and_processor(
    model_yaml: str | Path,
    load_pretrained_components: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor]:
    """Build model and processor from a model YAML, resizing embeddings if needed."""

    model_data = load_yaml(model_yaml)
    config = config_from_yaml_dict(model_data)
    processor = processor_from_yaml_dict(model_data, config)
    model = model_from_yaml_dict(
        model_data,
        config=config,
        load_pretrained_components=load_pretrained_components,
        model_kwargs=_coerce_model_kwargs(model_kwargs),
    )

    tokenizer_vocab_size = len(processor.tokenizer)
    embedding_vocab_size = model.get_input_embeddings().num_embeddings
    if tokenizer_vocab_size > embedding_vocab_size:
        try:
            model.resize_token_embeddings(tokenizer_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(tokenizer_vocab_size)
        resized_vocab_size = model.get_input_embeddings().num_embeddings
        model.config.text_config.vocab_size = resized_vocab_size
        model.config.vocab_size = resized_vocab_size
    return model, processor


def _load_checkpoint_model_and_processor(
    model_checkpoint: str | Path,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor]:
    """Load a saved LLaVa-Anything checkpoint and its processor."""

    checkpoint = Path(model_checkpoint)
    processor = LlavaAnythingProcessor.from_pretrained(checkpoint)
    model = LlavaAnythingForConditionalGeneration.from_pretrained(
        checkpoint,
        **(_coerce_model_kwargs(model_kwargs) or {}),
    )
    return model, processor


def run_training_from_yaml(path: str | Path) -> LlavaPretrainingResult:
    """Run the full pretraining loop described by a YAML configuration."""

    data = _load_training_yaml(path)
    model_yaml = data.get("model_yaml")
    model_checkpoint = data.get("model_checkpoint")
    if bool(model_yaml) == bool(model_checkpoint):
        raise ValueError("Exactly one of model_yaml or model_checkpoint is required.")
    data_section = data.get("data", {})
    if not isinstance(data_section, dict):
        raise ValueError("data must be a mapping.")
    training_section = data.get("training", {})
    if not isinstance(training_section, dict):
        raise ValueError("training must be a mapping.")
    if "output_dir" not in training_section:
        raise ValueError("training.output_dir is required.")
    logging_section = data.get("logging", {}) or {}
    if not isinstance(logging_section, dict):
        raise ValueError("logging must be a mapping when provided.")
    wandb_section = data["wandb"] if "wandb" in data else None
    if wandb_section is None and "wandb" in data:
        wandb_section = {}
    if wandb_section is not None and not isinstance(wandb_section, dict):
        raise ValueError("wandb must be a mapping when provided.")
    configure_wandb(training_section, wandb_section)

    if model_checkpoint:
        model, processor = _load_checkpoint_model_and_processor(
            model_checkpoint,
            model_kwargs=data.get("model_kwargs"),
        )
    else:
        model, processor = _build_model_and_processor(
            model_yaml,
            load_pretrained_components=bool(data.get("load_pretrained_components", True)),
            model_kwargs=data.get("model_kwargs"),
        )
    model.config.use_cache = False
    if hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    trainable_modules = str(training_section.get("trainable_modules", "projector"))
    training_args_data = dict(training_section)
    training_args_data.pop("trainable_modules", None)
    model_max_length = training_args_data.pop("model_max_length", None)
    trainable_names = apply_trainable_modules(model, trainable_modules)
    available_images_cache_dir = data_section.get("available_images_cache_dir")
    if available_images_cache_dir is None and bool(
        data_section.get("available_images_cache", True)
    ):
        available_images_cache_dir = training_section["output_dir"]

    dataset = LlavaPretrainDataset(
        data_path=data_section["data_path"],
        image_folder=data_section["image_folder"],
        processor=processor,
        max_samples=data_section.get("max_samples"),
        available_images_only=bool(data_section.get("available_images_only", True)),
        available_images_cache_dir=available_images_cache_dir,
        refresh_available_images_cache=bool(
            data_section.get("refresh_available_images_cache", False)
        ),
        system_prompt=data_section.get("system_prompt"),
        model_max_length=model_max_length,
    )
    log_preview_samples(dataset, int(logging_section.get("preview_samples", 0)))
    collator = LlavaPretrainDataCollator(processor.tokenizer)
    training_args = _coerce_training_arguments(training_args_data)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    resume_from_checkpoint = _resolve_resume_from_checkpoint(training_args)
    if resume_from_checkpoint and _is_main_process():
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return LlavaPretrainingResult(
        train_result=train_result,
        output_dir=output_dir,
        trainable_parameter_names=trainable_names,
    )


def main() -> None:
    """CLI entry point for running LLaVa-Anything pretraining from YAML."""

    parser = argparse.ArgumentParser(description="Run LLaVa-Anything training from a YAML config.")
    parser.add_argument("training_yaml", type=Path)
    args = parser.parse_args()
    result = run_training_from_yaml(args.training_yaml)
    print(f"training_loss: {result.train_result.training_loss}")
    print(f"output_dir: {result.output_dir}")
    print(f"trainable_parameters: {len(result.trainable_parameter_names)}")


__all__ = [
    "IGNORE_INDEX",
    "LlavaPretrainDataCollator",
    "LlavaPretrainDataset",
    "LlavaPretrainingResult",
    "apply_trainable_modules",
    "configure_wandb",
    "log_preview_samples",
    "run_training_from_yaml",
]


if __name__ == "__main__":
    main()
