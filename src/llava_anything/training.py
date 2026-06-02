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

from .builder import config_from_yaml_dict, load_yaml, model_from_yaml_dict, processor_from_yaml_dict
from .modeling_llava_anything import LlavaAnythingForConditionalGeneration
from .processing_llava_anything import LlavaAnythingProcessor

IGNORE_INDEX = -100


def _load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def _role_name(raw_role: str) -> str:
    role = raw_role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role


def _conversation_text(record: dict[str, Any]) -> tuple[str, str]:
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
    user_text, assistant_text = _conversation_text(record)
    return _render_prefix(processor, user_text, system_prompt), assistant_text


def log_preview_samples(dataset: "LlavaPretrainDataset", count: int = 2) -> None:
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


def _tokenize_text(processor: LlavaAnythingProcessor, text: str) -> torch.LongTensor:
    encoded = processor(text=text, return_tensors="pt", add_special_tokens=False)
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
        system_prompt: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_folder = Path(image_folder)
        self.processor = processor
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string when provided.")
        self.system_prompt = system_prompt or None
        records = _load_json_records(self.data_path)
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
        return len(self.records)

    def _record_image_path(self, record: dict[str, Any]) -> Path:
        image_name = record.get("image")
        if not image_name:
            raise ValueError("Pretraining records must include an image path.")
        return self.image_folder / str(image_name)

    def _record_has_available_image(self, record: dict[str, Any]) -> bool:
        image_name = record.get("image")
        if not image_name:
            return False
        return (self.image_folder / str(image_name)).is_file()

    def _filter_available_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        available: list[dict[str, Any]] = []
        skipped_count = 0
        for record in records:
            if self._record_has_available_image(record):
                available.append(record)
            else:
                skipped_count += 1
        return available, skipped_count

    def _load_image(self, record: dict[str, Any]) -> Image.Image:
        image_path = self._record_image_path(record)
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        prefix, assistant_text = _preview_record(self.processor, record, self.system_prompt)
        eos = self.processor.tokenizer.eos_token or ""
        full_text = f"{prefix}{assistant_text}{eos}"

        image = self._load_image(record)
        if getattr(self.processor, "image_mode", "fixed") == "anyres":
            full_inputs = self.processor(images=image, text=full_text, return_tensors="pt", add_special_tokens=False)
            prefix_inputs = self.processor(images=image, text=prefix, return_tensors="pt", add_special_tokens=False)
            input_ids = full_inputs["input_ids"][0]
            prefix_ids = prefix_inputs["input_ids"][0]
            image_inputs = full_inputs
        else:
            input_ids = _tokenize_text(self.processor, full_text)
            prefix_ids = _tokenize_text(self.processor, prefix)
            image_inputs = self.processor(images=image, return_tensors="pt")

        labels = input_ids.clone()
        labels[: prefix_ids.shape[0]] = IGNORE_INDEX
        labels[input_ids == self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)] = IGNORE_INDEX

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": image_inputs["pixel_values"][0],
        }
        if "image_sizes" in image_inputs:
            sample["image_sizes"] = image_inputs["image_sizes"][0]
        return sample


@dataclass
class LlavaPretrainDataCollator:
    tokenizer: Any

    def _pad_sequence(self, tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        if getattr(self.tokenizer, "padding_side", "right") == "left":
            tensors = [torch.flip(tensor, dims=[0]) for tensor in tensors]
            padded = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
            return torch.flip(padded, dims=[1])
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids = self._pad_sequence([instance["input_ids"] for instance in instances], pad_token_id)
        labels = self._pad_sequence([instance["labels"] for instance in instances], IGNORE_INDEX)
        attention_mask = input_ids.ne(pad_token_id)
        pixel_value_tensors = [instance["pixel_values"] for instance in instances]
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
            pixel_values = torch.stack(padded_pixel_values)
        else:
            pixel_values = torch.stack(pixel_value_tensors)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
        if "image_sizes" in instances[0]:
            batch["image_sizes"] = torch.stack([instance["image_sizes"] for instance in instances])
        return batch


def apply_trainable_modules(model: LlavaAnythingForConditionalGeneration, trainable_modules: str = "projector") -> list[str]:
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


def _load_training_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _coerce_torch_dtype(value: Any) -> Any:
    if not isinstance(value, str) or value == "auto":
        return value
    if hasattr(torch, value):
        dtype = getattr(torch, value)
        if isinstance(dtype, torch.dtype):
            return dtype
    return value


def _coerce_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
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
    checkpoint = Path(model_checkpoint)
    processor = LlavaAnythingProcessor.from_pretrained(checkpoint)
    model = LlavaAnythingForConditionalGeneration.from_pretrained(
        checkpoint,
        **(_coerce_model_kwargs(model_kwargs) or {}),
    )
    return model, processor


def run_pretraining_from_yaml(path: str | Path) -> LlavaPretrainingResult:
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
    trainable_names = apply_trainable_modules(model, trainable_modules)

    dataset = LlavaPretrainDataset(
        data_path=data_section["data_path"],
        image_folder=data_section["image_folder"],
        processor=processor,
        max_samples=data_section.get("max_samples"),
        available_images_only=bool(data_section.get("available_images_only", True)),
        system_prompt=data_section.get("system_prompt"),
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
    train_result = trainer.train()

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
    parser = argparse.ArgumentParser(description="Run LLaVa-Anything training from a YAML config.")
    parser.add_argument("training_yaml", type=Path)
    args = parser.parse_args()
    result = run_pretraining_from_yaml(args.training_yaml)
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
    "run_pretraining_from_yaml",
]


if __name__ == "__main__":
    main()
