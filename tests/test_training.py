from __future__ import annotations

import json
import os
import sys
from collections import deque
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

import llava_anything.training as training_module
from llava_anything.builder import save_from_yaml
from llava_anything.training import (
    IGNORE_INDEX,
    LlavaPretrainDataCollator,
    LlavaPretrainDataset,
    LlavaAnythingTrainer,
    apply_frozen_parameter_patterns,
    apply_trainable_modules,
    configure_wandb,
    log_preview_samples,
    run_pretraining_from_yaml,
    _coerce_training_arguments,
    _resolve_resume_from_checkpoint,
)


def _write_pretrain_json(path: Path, image_name: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "image": image_name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                }
            ]
        )
    )


def test_pretrain_dataset_reads_huggingface_dataset_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_buffer = BytesIO()
    tiny_image.save(image_buffer, format="PNG")
    records = [
        {
            "id": "decoded-image",
            "image": tiny_image.copy(),
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "decoded"},
            ],
        },
        {
            "id": "bytes-image",
            "image": {"bytes": image_buffer.getvalue(), "path": "sample.png"},
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "bytes"},
            ],
        },
    ]
    calls = []

    class FakeHFDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, indices):
            return FakeHFDataset([self.rows[index] for index in indices])

        def __iter__(self):
            return iter(self.rows)

    def fake_load_dataset(path, name=None, split=None, **kwargs):
        calls.append((path, name, split, kwargs))
        return FakeHFDataset(records)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    dataset = LlavaPretrainDataset(
        hf_dataset_path="naufalso/LLaVA-Pretrain",
        hf_dataset_name="pretrain",
        hf_dataset_split="train",
        image_folder=None,
        processor=processor,
        max_samples=2,
    )

    assert calls == [("naufalso/LLaVA-Pretrain", "pretrain", "train", {})]
    assert [record["id"] for record in dataset.records] == ["decoded-image", "bytes-image"]
    assert dataset[0]["pixel_values"].shape == (3, 8, 8)
    assert dataset[1]["pixel_values"].shape == (3, 8, 8)


def test_pretraining_yaml_passes_huggingface_dataset_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_dataset_kwargs = {}
    output_dir = tmp_path / "out"
    training_yaml = tmp_path / "train.yaml"
    training_yaml.write_text(
        yaml.safe_dump(
            {
                "model_checkpoint": "checkpoint",
                "data": {
                    "hf_dataset_path": "naufalso/LLaVA-Pretrain",
                    "hf_dataset_name": "pretrain",
                    "hf_dataset_split": "train",
                    "max_samples": 4,
                },
                "training": {
                    "output_dir": str(output_dir),
                    "max_steps": 0,
                    "report_to": [],
                    "remove_unused_columns": False,
                },
            }
        )
    )

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.language_model = SimpleNamespace(config=SimpleNamespace(use_cache=True))

        def named_parameters(self):
            return []

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeProcessor:
        tokenizer = SimpleNamespace()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeDataset:
        pass

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(training_loss=0.0)

    def fake_dataset(**kwargs):
        captured_dataset_kwargs.update(kwargs)
        return FakeDataset()

    fake_training_args = SimpleNamespace(output_dir=str(output_dir), resume_from_checkpoint=False)
    monkeypatch.setattr(
        training_module,
        "_load_checkpoint_model_and_processor",
        lambda *args, **kwargs: (FakeModel(), FakeProcessor()),
    )
    monkeypatch.setattr(training_module, "apply_trainable_modules", lambda *args, **kwargs: [])
    monkeypatch.setattr(training_module, "apply_frozen_parameter_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(training_module, "LlavaPretrainDataset", fake_dataset)
    monkeypatch.setattr(training_module, "LlavaPretrainDataCollator", lambda *args, **kwargs: object())
    monkeypatch.setattr(training_module, "_coerce_training_arguments", lambda *args, **kwargs: fake_training_args)
    monkeypatch.setattr(training_module, "_resolve_resume_from_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_module, "LlavaAnythingTrainer", FakeTrainer)

    training_module.run_pretraining_from_yaml(training_yaml)

    assert captured_dataset_kwargs["data_path"] is None
    assert captured_dataset_kwargs["image_folder"] is None
    assert captured_dataset_kwargs["hf_dataset_path"] == "naufalso/LLaVA-Pretrain"
    assert captured_dataset_kwargs["hf_dataset_name"] == "pretrain"
    assert captured_dataset_kwargs["hf_dataset_split"] == "train"
    assert captured_dataset_kwargs["max_samples"] == 4


def test_pretrain_dataset_reads_llava_json_and_masks_prompt_tokens(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)

    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    sample = dataset[0]

    assert sample["input_ids"].shape == sample["labels"].shape
    assert sample["pixel_values"].shape == (3, 8, 8)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_positions = sample["input_ids"] == image_token_id
    assert int(image_positions.sum().item()) == 4
    assert torch.all(sample["labels"][image_positions] == IGNORE_INDEX)
    assert int((sample["labels"] != IGNORE_INDEX).sum().item()) > 0


def test_pretrain_collator_pads_text_and_stacks_images(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)
    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    collator = LlavaPretrainDataCollator(processor.tokenizer)

    batch = collator([dataset[0], dataset[0]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["pixel_values"].shape == (2, 3, 8, 8)
    assert "_metadata" not in batch

    debug_batch = LlavaPretrainDataCollator(processor.tokenizer, include_metadata=True)([dataset[0]])
    assert debug_batch["_metadata"][0]["record_id"] == "sample-1"


def test_pretrain_dataset_supports_text_only_records(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    data_path = tmp_path / "text-only.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "text-only",
                    "conversations": [
                        {"from": "human", "value": "What is the capital of France?"},
                        {"from": "gpt", "value": "Paris"},
                    ],
                }
            ]
        )
    )

    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    sample = dataset[0]

    assert sample["input_ids"].shape == sample["labels"].shape
    assert "pixel_values" not in sample
    assert "image_sizes" not in sample
    assert int((sample["labels"] != IGNORE_INDEX).sum().item()) > 0


def test_pretrain_dataset_truncates_to_explicit_model_max_length(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    processor.tokenizer.model_max_length = 4
    data_path = tmp_path / "long-text.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "long-text",
                    "conversations": [
                        {"from": "human", "value": "token_2 token_3 token_4 token_5"},
                        {"from": "gpt", "value": "token_6 token_7 token_8 token_9 token_10 token_11"},
                    ],
                }
            ]
        )
    )

    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=tmp_path,
        processor=processor,
        model_max_length=7,
    )
    sample = dataset[0]

    assert dataset.model_max_length == 7
    assert sample["input_ids"].shape[0] == 7
    assert sample["labels"].shape == sample["input_ids"].shape


def test_pretrain_dataset_infers_model_max_length_from_tokenizer(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    processor.tokenizer.model_max_length = 6
    data_path = tmp_path / "long-text.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "long-text",
                    "conversations": [
                        {"from": "human", "value": "token_2 token_3 token_4 token_5"},
                        {"from": "gpt", "value": "token_6 token_7 token_8 token_9 token_10 token_11"},
                    ],
                }
            ]
        )
    )

    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    sample = dataset[0]

    assert dataset.model_max_length == 6
    assert sample["input_ids"].shape[0] == 6


def test_pretrain_collator_supports_mixed_text_and_image_records(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "mixed.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "text-only",
                    "conversations": [
                        {"from": "human", "value": "Say hello."},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
                {
                    "id": "image",
                    "image": image_path.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "available"},
                    ],
                },
            ]
        )
    )
    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    collator = LlavaPretrainDataCollator(processor.tokenizer)

    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["pixel_values"].shape == (1, 3, 8, 8)

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)
    output = model(**batch)
    assert output.loss >= 0



def test_pretrain_dataset_collates_anyres_images_with_image_sizes(
    tmp_path: Path,
    tiny_text_component_dir: Path,
    tiny_vision_component_dir: Path,
) -> None:
    model_yaml = tmp_path / "anyres-model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "image": {
                    "mode": "anyres",
                    "anyres": {"enabled": True, "grid_pinpoints": [[8, 8], [8, 16]]},
                },
                "text_model": {"name_or_path": str(tiny_text_component_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(tiny_vision_component_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )
    model_dir = tmp_path / "anyres-model"
    save_from_yaml(model_yaml, model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    square_path = tmp_path / "square.jpg"
    wide_path = tmp_path / "wide.jpg"

    Image.new("RGB", (8, 8)).save(square_path)
    Image.new("RGB", (16, 8)).save(wide_path)
    data_path = tmp_path / "anyres.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "square",
                    "image": square_path.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
                {
                    "id": "wide",
                    "image": wide_path.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
            ]
        )
    )
    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    collator = LlavaPretrainDataCollator(processor.tokenizer)

    square = dataset[0]
    wide = dataset[1]
    batch = collator([square, wide])

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert square["image_sizes"].tolist() == [8, 8]
    assert wide["image_sizes"].tolist() == [8, 16]
    assert square["pixel_values"].shape == (2, 3, 8, 8)
    assert wide["pixel_values"].shape == (3, 3, 8, 8)
    assert batch["pixel_values"].shape == (2, 3, 3, 8, 8)
    assert batch["image_sizes"].tolist() == [[8, 8], [8, 16]]
    assert (batch["input_ids"] == image_token_id).sum(dim=1).tolist() == [10, 14]


def test_pretrain_dataset_skips_truncated_anyres_image_tokens(
    tmp_path: Path,
    tiny_text_component_dir: Path,
    tiny_vision_component_dir: Path,
) -> None:
    model_yaml = tmp_path / "anyres-model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "image": {
                    "mode": "anyres",
                    "anyres": {"enabled": True, "grid_pinpoints": [[8, 8], [8, 16]]},
                },
                "text_model": {"name_or_path": str(tiny_text_component_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(tiny_vision_component_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )
    model_dir = tmp_path / "anyres-model"
    save_from_yaml(model_yaml, model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    square_path = tmp_path / "square.jpg"
    wide_path = tmp_path / "wide.jpg"
    Image.new("RGB", (8, 8)).save(square_path)
    Image.new("RGB", (16, 8)).save(wide_path)
    data_path = tmp_path / "anyres.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "truncated-image",
                    "image": wide_path.name,
                    "conversations": [
                        {"from": "human", "value": "token_2 token_3 token_4 token_5 token_6 token_7 <image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
                {
                    "id": "usable-image",
                    "image": square_path.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                }
            ]
        )
    )
    with pytest.warns(UserWarning, match="1 sample skipped before training"):
        dataset = LlavaPretrainDataset(
            data_path=data_path,
            image_folder=tmp_path,
            processor=processor,
            model_max_length=12,
            image_token_mismatch_prefilter=True,
        )

    sample = dataset[0]

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert sample is not None
    assert [record["id"] for record in dataset.records] == ["usable-image"]
    assert int((sample["input_ids"] == image_token_id).sum().item()) == 10


def test_pretrain_dataset_lazily_skips_invalid_image_token_records_by_default(
    tmp_path: Path,
    tiny_text_component_dir: Path,
    tiny_vision_component_dir: Path,
) -> None:
    model_yaml = tmp_path / "anyres-model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "image": {
                    "mode": "anyres",
                    "anyres": {"enabled": True, "grid_pinpoints": [[8, 8], [8, 16]]},
                },
                "text_model": {"name_or_path": str(tiny_text_component_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(tiny_vision_component_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )
    model_dir = tmp_path / "anyres-model"
    save_from_yaml(model_yaml, model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    square_path = tmp_path / "square.jpg"
    wide_path = tmp_path / "wide.jpg"
    Image.new("RGB", (8, 8)).save(square_path)
    Image.new("RGB", (16, 8)).save(wide_path)
    data_path = tmp_path / "anyres.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "runtime-skip",
                    "image": wide_path.name,
                    "conversations": [
                        {"from": "human", "value": "token_2 token_3 token_4 token_5 token_6 token_7 <image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
                {
                    "id": "usable-image",
                    "image": square_path.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
            ]
        )
    )

    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=tmp_path,
        processor=processor,
        model_max_length=12,
    )
    sample = dataset[0]

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert len(dataset) == 2
    assert [record["id"] for record in dataset.records] == ["runtime-skip", "usable-image"]
    assert int((sample["input_ids"] == image_token_id).sum().item()) == 10


def test_pretrain_dataset_lazily_skips_text_only_records_with_image_token(
    tmp_path: Path,
    tiny_text_component_dir: Path,
    tiny_vision_component_dir: Path,
) -> None:
    model_yaml = tmp_path / "anyres-model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "image": {
                    "mode": "anyres",
                    "anyres": {"enabled": True, "grid_pinpoints": [[8, 8], [8, 16]]},
                },
                "text_model": {"name_or_path": str(tiny_text_component_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(tiny_vision_component_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )
    model_dir = tmp_path / "anyres-model"
    save_from_yaml(model_yaml, model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    data_path = tmp_path / "anyres.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "malformed-text",
                    "conversations": [
                        {"from": "human", "value": "Describe this.\n<image>"},
                        {"from": "gpt", "value": "bad"},
                    ],
                },
                {
                    "id": "usable-text",
                    "conversations": [
                        {"from": "human", "value": "Say hello"},
                        {"from": "gpt", "value": "hello"},
                    ],
                },
            ]
        )
    )

    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    with pytest.warns(UserWarning, match="text-only record.*image token"):
        sample = dataset[0]

    assert sample["_metadata"]["record_id"] == "usable-text"
    assert "pixel_values" not in sample
    assert int((sample["labels"] != IGNORE_INDEX).sum().item()) > 0


def test_pretrain_dataset_truncates_without_dropping_all_targets(
    tmp_path: Path,
    tiny_text_component_dir: Path,
    tiny_vision_component_dir: Path,
) -> None:
    model_yaml = tmp_path / "anyres-model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "image": {
                    "mode": "anyres",
                    "anyres": {"enabled": True, "grid_pinpoints": [[8, 8]]},
                },
                "text_model": {"name_or_path": str(tiny_text_component_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(tiny_vision_component_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )
    model_dir = tmp_path / "anyres-model"
    save_from_yaml(model_yaml, model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(model_dir)
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    data_path = tmp_path / "anyres.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "long-prompt",
                    "image": image_path.name,
                    "conversations": [
                        {"from": "human", "value": "token_2 token_3 token_4 token_5 token_6 token_7 <image>\nWhat is shown?"},
                        {"from": "gpt", "value": "hello"},
                    ],
                }
            ]
        )
    )

    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=tmp_path,
        processor=processor,
        model_max_length=12,
    )
    sample = dataset[0]

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert sample["input_ids"].shape[0] == 12
    assert int((sample["input_ids"] == image_token_id).sum().item()) == 10
    assert int((sample["labels"] != IGNORE_INDEX).sum().item()) > 0


def test_pretrain_collator_rejects_batches_without_targets(
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    collator = LlavaPretrainDataCollator(processor.tokenizer)

    with pytest.raises(ValueError, match="no supervised target tokens"):
        collator(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.full((3,), IGNORE_INDEX),
                }
            ]
        )


def test_pretrain_dataset_can_filter_records_to_text_only_and_available_images(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    available_image = tmp_path / "available.jpg"
    tiny_image.save(available_image)
    data_path = tmp_path / "instruct.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "text-only",
                    "conversations": [
                        {"from": "human", "value": "What is shown?"},
                        {"from": "gpt", "value": "no image"},
                    ],
                },
                {
                    "id": "missing",
                    "image": "missing.jpg",
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "missing"},
                    ],
                },
                {
                    "id": "available",
                    "image": available_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "available"},
                    ],
                },
            ]
        )
    )

    with pytest.warns(UserWarning, match="1 image not found and skipping those"):
        dataset = LlavaPretrainDataset(
            data_path=data_path,
            image_folder=tmp_path,
            processor=processor,
            available_images_only=True,
        )

    assert len(dataset) == 2
    assert [record["id"] for record in dataset.records] == ["text-only", "available"]
    assert "pixel_values" not in dataset[0]
    assert dataset[1]["pixel_values"].shape == (3, 8, 8)

    with pytest.warns(UserWarning) as warning_records:
        image_only_dataset = LlavaPretrainDataset(
            data_path=data_path,
            image_folder=tmp_path,
            processor=processor,
            available_images_only=True,
            require_image=True,
        )
    warning_messages = "\n".join(str(record.message) for record in warning_records)
    assert "1 text-only record skipped" in warning_messages
    assert "1 image not found" in warning_messages
    assert len(image_only_dataset) == 1
    assert [record["id"] for record in image_only_dataset.records] == ["available"]


def test_pretrain_dataset_lazily_skips_images_failing_constraints(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    small_image = tmp_path / "small.jpg"
    large_image = tmp_path / "large.jpg"
    tiny_image.save(small_image)
    Image.new("RGB", (16, 16)).save(large_image)
    data_path = tmp_path / "instruct.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "too-small",
                    "image": small_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "small"},
                    ],
                },
                {
                    "id": "large-enough",
                    "image": large_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "large"},
                    ],
                },
            ]
        )
    )
    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=tmp_path,
        processor=processor,
        min_image_width=16,
        min_image_height=16,
    )

    with pytest.warns(UserWarning, match="image constraints failed"):
        sample = dataset[0]

    assert sample["pixel_values"].shape == (3, 8, 8)


def test_pretrain_dataset_prefilters_images_failing_constraints(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    small_image = tmp_path / "small.jpg"
    large_image = tmp_path / "large.jpg"
    tiny_image.save(small_image)
    Image.new("RGB", (16, 16)).save(large_image)
    data_path = tmp_path / "instruct.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "too-small",
                    "image": small_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "small"},
                    ],
                },
                {
                    "id": "large-enough",
                    "image": large_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "large"},
                    ],
                },
            ]
        )
    )

    with pytest.warns(UserWarning, match="configured image constraints failed"):
        dataset = LlavaPretrainDataset(
            data_path=data_path,
            image_folder=tmp_path,
            processor=processor,
            available_images_cache_dir=tmp_path,
            min_image_width=16,
            min_image_height=16,
            image_constraint_prefilter=True,
        )

    assert len(dataset) == 1
    assert dataset.records[0]["id"] == "large-enough"
    assert (tmp_path / "skipped_image_constraint_indices.json").is_file()


def test_pretrain_dataset_warns_and_skips_missing_images_by_default(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    available_image = tmp_path / "available.jpg"
    tiny_image.save(available_image)
    data_path = tmp_path / "instruct.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "missing",
                    "image": "missing.jpg",
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "missing"},
                    ],
                },
                {
                    "id": "available",
                    "image": available_image.name,
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is shown?"},
                        {"from": "gpt", "value": "available"},
                    ],
                },
            ]
        )
    )

    with pytest.warns(UserWarning, match="1 image not found and skipping those"):
        dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)

    assert len(dataset) == 1
    assert dataset.records[0]["id"] == "available"


def test_pretraining_from_yaml_can_resume_from_composed_checkpoint(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "stage2.json"
    _write_pretrain_json(data_path, image_path.name)
    output_dir = tmp_path / "stage2-trained"
    training_yaml = tmp_path / "stage2.yaml"
    training_yaml.write_text(
        yaml.safe_dump(
            {
                "model_checkpoint": str(tiny_full_model_dir),
                "data": {
                    "data_path": str(data_path),
                    "image_folder": str(tmp_path),
                    "available_images_only": True,
                },
                "training": {
                    "output_dir": str(output_dir),
                    "trainable_modules": "full",
                    "max_steps": 1,
                    "per_device_train_batch_size": 1,
                    "learning_rate": 1.0e-5,
                    "save_strategy": "no",
                    "report_to": [],
                    "remove_unused_columns": False,
                    "seed": 0,
                },
            }
        )
    )

    result = run_pretraining_from_yaml(training_yaml)

    assert result.train_result.training_loss >= 0
    assert (output_dir / "config.json").exists()
    assert any(name.startswith("language_model.") for name in result.trainable_parameter_names)
    assert any(name.startswith("vision_tower.") for name in result.trainable_parameter_names)
    assert any(name.startswith("multi_modal_projector.") for name in result.trainable_parameter_names)


def test_projector_only_freezes_language_and_vision(tiny_model_yaml_path: Path, tiny_full_model_dir: Path) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)

    trainable = apply_trainable_modules(model, "projector")

    assert trainable
    assert all(name.startswith("multi_modal_projector.") for name in trainable)
    assert not any(param.requires_grad for param in model.language_model.parameters())
    assert not any(param.requires_grad for param in model.vision_tower.parameters())
    assert all(param.requires_grad for param in model.multi_modal_projector.parameters())


def test_frozen_parameter_patterns_remove_matching_trainable_parameters(
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)
    apply_trainable_modules(model, "full")

    frozen = apply_frozen_parameter_patterns(model, ["multi_modal_projector.*"])

    assert frozen
    assert all(name.startswith("multi_modal_projector.") for name in frozen)
    assert not any(param.requires_grad for param in model.multi_modal_projector.parameters())
    assert any(param.requires_grad for param in model.language_model.parameters())


def test_nonfinite_parameter_check_skips_unavailable_zero3_placeholders() -> None:
    class FakeZeroStatus:
        name = "NOT_AVAILABLE"

    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(float("nan"))
    model.weight.ds_status = FakeZeroStatus()

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = model
    trainer.accelerator = None

    assert trainer._local_nonfinite_parameter_summaries() == []


def test_nonfinite_parameter_check_keeps_available_zero3_parameters() -> None:
    class FakeZeroStatus:
        name = "AVAILABLE"

    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(float("nan"))
    model.weight.ds_status = FakeZeroStatus()

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = model
    trainer.accelerator = None

    summaries = trainer._local_nonfinite_parameter_summaries()

    assert summaries
    assert summaries[0].startswith("weight:")


def test_nonfinite_parameter_check_ignores_zero3_visible_param_when_master_is_finite() -> None:
    class FakeZeroStatus:
        name = "AVAILABLE"

    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.fp32_partitioned_groups_flat = [torch.ones(4, dtype=torch.float32)]

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()

    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(float("nan"))
    model.weight.ds_status = FakeZeroStatus()

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = model
    trainer.accelerator = type(
        "Accelerator",
        (),
        {"deepspeed_engine_wrapped": type("Wrapped", (), {"engine": FakeEngine()})()},
    )()

    assert trainer._local_nonfinite_parameter_summaries() == []


def test_zero3_visible_nonfinite_parameters_are_diagnostic_only() -> None:
    class FakeZeroStatus:
        name = "AVAILABLE"

    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.fp32_partitioned_groups_flat = [torch.ones(4, dtype=torch.float32)]

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()

    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(float("nan"))
    model.weight.ds_status = FakeZeroStatus()

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = model
    trainer.accelerator = type(
        "Accelerator",
        (),
        {"deepspeed_engine_wrapped": type("Wrapped", (), {"engine": FakeEngine()})()},
    )()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer._visible_zero3_nonfinite_parameter_warning_count = 0

    with pytest.warns(UserWarning, match="visible ZeRO-3 model parameter views"):
        trainer._warn_if_visible_zero3_parameters_nonfinite()

    bad, message = trainer._any_rank_has_nonfinite_training_state()

    assert not bad
    assert message == "no local non-finite training state"


def test_skipped_loss_like_is_scalar_with_grad_fn() -> None:
    loss = torch.tensor(float("nan"), requires_grad=True)

    skipped_loss = LlavaAnythingTrainer._skipped_loss_like(loss)

    assert skipped_loss.numel() == 1
    assert skipped_loss.grad_fn is not None
    assert torch.isfinite(skipped_loss)
    skipped_loss.backward()


def test_nonfinite_loss_guard_skips_optimizer_step() -> None:
    class DummyOptimizer:
        def __init__(self) -> None:
            self.called = False
            self._is_overflow = False

        def step(self) -> None:
            self.called = True

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.optimizer = DummyOptimizer()
    trainer._wrapped_optimizer = None
    trainer._skip_next_optimizer_step = True
    trainer._optimizer_step_skip_marked_by_guard = False

    trainer._wrap_optimizer_step_if_needed()
    trainer.optimizer.step()

    assert not trainer.optimizer.called
    assert trainer.optimizer._is_overflow is True
    assert trainer._skip_next_optimizer_step is False
    assert trainer._optimizer_step_skip_marked_by_guard is True


def test_nonfinite_loss_guard_preserves_external_overflow_flag() -> None:
    class DummyOptimizer:
        def __init__(self) -> None:
            self.called = False
            self._is_overflow = True

        def step(self) -> None:
            self.called = True

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.optimizer = DummyOptimizer()
    trainer._wrapped_optimizer = None
    trainer._skip_next_optimizer_step = False
    trainer._optimizer_step_skip_marked_by_guard = False

    trainer._wrap_optimizer_step_if_needed()
    trainer.optimizer.step()

    assert trainer.optimizer.called
    assert trainer.optimizer._is_overflow is True


def test_nonfinite_loss_guard_clears_only_its_own_previous_skip_flag() -> None:
    class DummyOptimizer:
        def __init__(self) -> None:
            self.called = False
            self._is_overflow = True

        def step(self) -> None:
            self.called = True

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.optimizer = DummyOptimizer()
    trainer._wrapped_optimizer = None
    trainer._skip_next_optimizer_step = False
    trainer._optimizer_step_skip_marked_by_guard = True

    trainer._wrap_optimizer_step_if_needed()
    trainer.optimizer.step()

    assert trainer.optimizer.called
    assert trainer.optimizer._is_overflow is False
    assert trainer._optimizer_step_skip_marked_by_guard is False


def test_nonfinite_gradient_norm_guard_detects_any_rank_nonfinite_norm() -> None:
    trainer = object.__new__(LlavaAnythingTrainer)

    assert not trainer._any_rank_should_skip_gradient_norm(torch.tensor(1.0))
    assert trainer._any_rank_should_skip_gradient_norm(torch.tensor(float("nan")))


def test_nonfinite_parameter_check_reports_bad_parameter() -> None:
    trainer = object.__new__(LlavaAnythingTrainer)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    trainer.model = model
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer._optimizer_step_attempts = 7
    trainer._recent_batch_metadata = deque(maxlen=5)

    with pytest.raises(RuntimeError, match="Non-finite model parameter detected after optimizer step 7"):
        trainer._raise_if_any_rank_has_nonfinite_parameter()


def test_nonfinite_parameter_check_reports_zero3_master_parameter() -> None:
    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.fp32_partitioned_groups_flat = [torch.tensor([1.0, float("nan")])]

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.accelerator = type(
        "Accelerator",
        (),
        {"deepspeed_engine_wrapped": type("Wrapped", (), {"engine": FakeEngine()})()},
    )()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer._optimizer_step_attempts = 3
    trainer._recent_batch_metadata = deque(maxlen=5)

    with pytest.raises(RuntimeError, match="deepspeed.fp32_partitioned_groups_flat"):
        trainer._raise_if_any_rank_has_nonfinite_parameter()


def test_nonfinite_training_state_check_reports_buffer() -> None:
    trainer = object.__new__(LlavaAnythingTrainer)
    model = torch.nn.BatchNorm1d(2)
    with torch.no_grad():
        model.running_mean[0] = float("inf")
    trainer.model = model
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.accelerator = None

    bad, message = trainer._any_rank_has_nonfinite_training_state()

    assert bad
    assert "running_mean" in message


def test_nonfinite_training_state_check_reports_optimizer_state() -> None:
    class FakeOptimizer:
        def __init__(self) -> None:
            self.state = {object(): {"exp_avg": torch.tensor([1.0, float("nan")])}}

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = FakeOptimizer()
    trainer.accelerator = None
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()

    bad, message = trainer._any_rank_has_nonfinite_training_state()

    assert bad
    assert "trainer.optimizer.state" in message
    assert "exp_avg" in message


def test_nonfinite_loss_persistence_counts_optimizer_windows() -> None:
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.max_consecutive_nonfinite_losses = 2
    trainer._consecutive_nonfinite_loss_windows = 0

    assert not trainer._record_skipped_optimizer_window("this accumulation window had a non-finite loss")
    assert trainer._consecutive_nonfinite_loss_windows == 1
    assert trainer._record_skipped_optimizer_window("this accumulation window had a non-finite loss")
    assert trainer._consecutive_nonfinite_loss_windows == 2


def test_deepspeed_backward_guard_skips_requested_step() -> None:
    class DummyTrainerOptimizer:
        def __init__(self) -> None:
            self._is_overflow = False

    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.zeroed = False
            self.reset = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zeroed = set_to_none

        def reset_cpu_buffers(self) -> None:
            self.reset = True

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()
            self.step_calls = 0
            self.backward_calls = 0
            self.zero_grad_calls = 0

        def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
            self.is_boundary = is_boundary

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            self.backward_calls += 1

        def step(self) -> None:
            self.step_calls += 1

        def zero_grad(self) -> None:
            self.zero_grad_calls += 1

    class FakeWrappedEngine:
        def __init__(self) -> None:
            self.engine = FakeEngine()

        def backward(self, loss: torch.Tensor, sync_gradients: bool = True, **kwargs) -> None:
            raise AssertionError("original wrapper should be replaced")

    wrapped = FakeWrappedEngine()
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.accelerator = type("Accelerator", (), {"deepspeed_engine_wrapped": wrapped})()
    trainer.optimizer = DummyTrainerOptimizer()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.skip_nonfinite_gradients = True
    trainer.nonfinite_gradient_steps = 0
    trainer._skip_next_optimizer_step = True
    trainer._optimizer_step_skip_reason = "test unsafe window"
    trainer._optimizer_step_skip_marked_by_guard = False
    trainer._wrapped_deepspeed_backward = None
    trainer._recent_batch_metadata = deque(maxlen=5)

    trainer._wrap_deepspeed_backward_if_needed()
    with pytest.warns(UserWarning, match="Skipping DeepSpeed optimizer step because test unsafe window"):
        wrapped.backward(torch.tensor(1.0, requires_grad=True), sync_gradients=True)

    assert wrapped.engine.backward_calls == 1
    assert wrapped.engine.step_calls == 0
    assert wrapped.engine.optimizer.zeroed is True
    assert wrapped.engine.optimizer.reset is True
    assert wrapped.engine.zero_grad_calls == 1
    assert trainer._skip_next_optimizer_step is False
    assert trainer.optimizer._is_overflow is True


def test_deepspeed_backward_guard_uses_overflow_cleanup_for_nonfinite_loss_window() -> None:
    class DummyTrainerOptimizer:
        def __init__(self) -> None:
            self._is_overflow = False

    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.overflow = False
            self.zeroed = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zeroed = set_to_none

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()
            self.step_calls = 0
            self.saw_overflow_during_step = False
            self.backward_calls = 0

        def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
            self.is_boundary = is_boundary

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            self.backward_calls += 1

        def step(self) -> None:
            self.step_calls += 1
            self.saw_overflow_during_step = self.optimizer.overflow
            self.optimizer.zero_grad(set_to_none=True)

    class FakeWrappedEngine:
        def __init__(self) -> None:
            self.engine = FakeEngine()

        def backward(self, loss: torch.Tensor, sync_gradients: bool = True, **kwargs) -> None:
            raise AssertionError("original wrapper should be replaced")

    wrapped = FakeWrappedEngine()
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.accelerator = type("Accelerator", (), {"deepspeed_engine_wrapped": wrapped})()
    trainer.optimizer = DummyTrainerOptimizer()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.skip_nonfinite_gradients = True
    trainer.nonfinite_gradient_steps = 0
    trainer.max_consecutive_nonfinite_losses = 8
    trainer._consecutive_nonfinite_loss_windows = 0
    trainer._skip_next_optimizer_step = True
    trainer._optimizer_step_skip_reason = "this accumulation window had a non-finite loss"
    trainer._optimizer_step_skip_marked_by_guard = False
    trainer._wrapped_deepspeed_backward = None
    trainer._recent_batch_metadata = deque(maxlen=5)

    trainer._wrap_deepspeed_backward_if_needed()
    with pytest.warns(UserWarning, match="Skipping DeepSpeed optimizer step because this accumulation window"):
        wrapped.backward(torch.tensor(1.0, requires_grad=True), sync_gradients=True)

    assert wrapped.engine.backward_calls == 1
    assert wrapped.engine.step_calls == 1
    assert wrapped.engine.saw_overflow_during_step is True
    assert wrapped.engine.optimizer.overflow is False
    assert wrapped.engine.optimizer.zeroed is True
    assert trainer._consecutive_nonfinite_loss_windows == 1
    assert trainer._skip_next_optimizer_step is False


def test_deepspeed_backward_guard_skips_nonfinite_zero3_gradient_state() -> None:
    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.norm_for_param_grads = {123: torch.tensor(float("nan"))}
            self.flat_grad_param = torch.nn.Parameter(torch.ones(2))
            self.flat_grad_param.grad = torch.full((2,), float("nan"))
            self.fp32_partitioned_groups_flat = [self.flat_grad_param]
            self.zeroed = False
            self.reset = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zeroed = set_to_none

        def reset_cpu_buffers(self) -> None:
            self.reset = True

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()
            self.step_calls = 0

        def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
            self.is_boundary = is_boundary

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            pass

        def step(self) -> None:
            self.step_calls += 1

        def zero_grad(self) -> None:
            pass

    class FakeWrappedEngine:
        def __init__(self) -> None:
            self.engine = FakeEngine()

        def backward(self, loss: torch.Tensor, sync_gradients: bool = True, **kwargs) -> None:
            raise AssertionError("original wrapper should be replaced")

    wrapped = FakeWrappedEngine()
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.accelerator = type("Accelerator", (), {"deepspeed_engine_wrapped": wrapped})()
    trainer.optimizer = type("Optimizer", (), {"_is_overflow": False})()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.skip_nonfinite_gradients = True
    trainer.nonfinite_gradient_steps = 0
    trainer._skip_next_optimizer_step = False
    trainer._optimizer_step_skip_reason = "this accumulation window was marked unsafe"
    trainer._optimizer_step_skip_marked_by_guard = False
    trainer._wrapped_deepspeed_backward = None
    trainer._recent_batch_metadata = deque(maxlen=5)

    trainer._wrap_deepspeed_backward_if_needed()
    with pytest.warns(UserWarning, match="gradient state is non-finite"):
        wrapped.backward(torch.tensor(1.0, requires_grad=True), sync_gradients=True)

    assert wrapped.engine.step_calls == 0
    assert wrapped.engine.optimizer.zeroed is True
    assert wrapped.engine.optimizer.reset is True
    assert wrapped.engine.optimizer.flat_grad_param.grad is not None
    assert torch.all(wrapped.engine.optimizer.flat_grad_param.grad == 0)
    assert trainer.nonfinite_gradient_steps == 1
    assert trainer._skip_next_optimizer_step is False


def test_deepspeed_backward_guard_skips_nonfinite_zero3_averaged_gradients() -> None:
    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.norm_for_param_grads = {}
            self.averaged_gradients = {0: [torch.tensor([1.0, float("nan")])]}
            self.fp32_partitioned_groups_flat = []
            self.zeroed = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zeroed = set_to_none

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()
            self.step_calls = 0

        def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
            self.is_boundary = is_boundary

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            pass

        def step(self) -> None:
            self.step_calls += 1

        def zero_grad(self) -> None:
            pass

    class FakeWrappedEngine:
        def __init__(self) -> None:
            self.engine = FakeEngine()

        def backward(self, loss: torch.Tensor, sync_gradients: bool = True, **kwargs) -> None:
            raise AssertionError("original wrapper should be replaced")

    wrapped = FakeWrappedEngine()
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.accelerator = type("Accelerator", (), {"deepspeed_engine_wrapped": wrapped})()
    trainer.optimizer = type("Optimizer", (), {"_is_overflow": False})()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.skip_nonfinite_gradients = True
    trainer.nonfinite_gradient_steps = 0
    trainer._skip_next_optimizer_step = False
    trainer._optimizer_step_skip_reason = "this accumulation window was marked unsafe"
    trainer._optimizer_step_skip_marked_by_guard = False
    trainer._wrapped_deepspeed_backward = None
    trainer._recent_batch_metadata = deque(maxlen=5)

    trainer._wrap_deepspeed_backward_if_needed()
    with pytest.warns(UserWarning, match="deepspeed.averaged_gradients"):
        wrapped.backward(torch.tensor(1.0, requires_grad=True), sync_gradients=True)

    assert wrapped.engine.step_calls == 0
    assert wrapped.engine.optimizer.zeroed is True
    assert wrapped.engine.optimizer.averaged_gradients == {}
    assert trainer.nonfinite_gradient_steps == 1


def test_deepspeed_backward_guard_skips_nonfinite_zero3_partition_buffer() -> None:
    class FakeZeroOptimizer:
        def __init__(self) -> None:
            self.norm_for_param_grads = {}
            self.grad_partitions_flat_buffer = torch.tensor([1.0, float("inf")])
            self.fp32_partitioned_groups_flat = []
            self.zeroed = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zeroed = set_to_none

    class FakeEngine:
        def __init__(self) -> None:
            self.optimizer = FakeZeroOptimizer()
            self.step_calls = 0

        def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
            self.is_boundary = is_boundary

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            pass

        def step(self) -> None:
            self.step_calls += 1

        def zero_grad(self) -> None:
            pass

    class FakeWrappedEngine:
        def __init__(self) -> None:
            self.engine = FakeEngine()

        def backward(self, loss: torch.Tensor, sync_gradients: bool = True, **kwargs) -> None:
            raise AssertionError("original wrapper should be replaced")

    wrapped = FakeWrappedEngine()
    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.accelerator = type("Accelerator", (), {"deepspeed_engine_wrapped": wrapped})()
    trainer.optimizer = type("Optimizer", (), {"_is_overflow": False})()
    trainer.args = type("Args", (), {"device": torch.device("cpu")})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.skip_nonfinite_gradients = True
    trainer.nonfinite_gradient_steps = 0
    trainer._skip_next_optimizer_step = False
    trainer._optimizer_step_skip_reason = "this accumulation window was marked unsafe"
    trainer._optimizer_step_skip_marked_by_guard = False
    trainer._wrapped_deepspeed_backward = None
    trainer._recent_batch_metadata = deque(maxlen=5)

    trainer._wrap_deepspeed_backward_if_needed()
    with pytest.warns(UserWarning, match="deepspeed.grad_partitions_flat_buffer"):
        wrapped.backward(torch.tensor(1.0, requires_grad=True), sync_gradients=True)

    assert wrapped.engine.step_calls == 0
    assert wrapped.engine.optimizer.zeroed is True
    assert torch.all(wrapped.engine.optimizer.grad_partitions_flat_buffer == 0)
    assert trainer.nonfinite_gradient_steps == 1


def test_nonfinite_loss_guard_ignores_deepspeed_dummy_optimizer() -> None:
    class DummyOptim:
        pass

    trainer = object.__new__(LlavaAnythingTrainer)
    trainer.optimizer = DummyOptim()
    trainer._wrapped_optimizer = None

    trainer._wrap_optimizer_step_if_needed()

    assert trainer._wrapped_optimizer is None


def test_pretraining_from_yaml_runs_projector_only_tiny_training(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_image,
) -> None:
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)
    output_dir = tmp_path / "trained"
    training_yaml = tmp_path / "pretrain.yaml"
    training_yaml.write_text(
        yaml.safe_dump(
            {
                "model_yaml": str(tiny_model_yaml_path),
                "data": {"data_path": str(data_path), "image_folder": str(tmp_path)},
                "training": {
                    "output_dir": str(output_dir),
                    "trainable_modules": "projector",
                    "model_max_length": 32,
                    "max_steps": 1,
                    "per_device_train_batch_size": 1,
                    "learning_rate": 1.0e-3,
                    "save_strategy": "no",
                    "report_to": [],
                    "remove_unused_columns": False,
                    "seed": 0,
                },
            }
        )
    )

    result = run_pretraining_from_yaml(training_yaml)

    assert result.train_result.training_loss >= 0
    assert (output_dir / "config.json").exists()
    assert result.trainable_parameter_names
    assert all(name.startswith("multi_modal_projector.") for name in result.trainable_parameter_names)


def test_projector_training_reduces_loss_on_tiny_synthetic_batch(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_image,
) -> None:
    from llava_anything.builder import config_from_yaml_dict, load_yaml, model_from_yaml_dict, processor_from_yaml_dict

    torch.manual_seed(0)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)

    model_data = load_yaml(tiny_model_yaml_path)
    config = config_from_yaml_dict(model_data)
    processor = processor_from_yaml_dict(model_data, config)
    model = model_from_yaml_dict(model_data, config, load_pretrained_components=True)
    apply_trainable_modules(model, "projector")
    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)
    batch = LlavaPretrainDataCollator(processor.tokenizer)([dataset[0]])
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.1)

    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        output = model(**batch)
        losses.append(float(output.loss.detach()))
        output.loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


def test_preview_sample_logging_shows_rendered_input_and_expected_output(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
    capsys,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)
    dataset = LlavaPretrainDataset(data_path=data_path, image_folder=tmp_path, processor=processor)

    log_preview_samples(dataset, count=1)

    captured = capsys.readouterr().out
    assert "Sample 0" in captured
    assert "Rendered input:" in captured
    assert "<image>" in captured
    assert "Expected output:" in captured
    assert "hello" in captured


def test_pretrain_dataset_includes_custom_system_prompt_in_rendered_input(
    tmp_path: Path,
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
    tiny_image,
    capsys,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)
    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    image_path = tmp_path / "image.jpg"
    tiny_image.save(image_path)
    data_path = tmp_path / "pretrain.json"
    _write_pretrain_json(data_path, image_path.name)
    system_prompt = "You are a careful visual assistant."
    dataset = LlavaPretrainDataset(
        data_path=data_path,
        image_folder=tmp_path,
        processor=processor,
        system_prompt=system_prompt,
    )

    log_preview_samples(dataset, count=1)

    captured = capsys.readouterr().out
    assert system_prompt in captured
    assert captured.index(system_prompt) < captured.index("<image>")


def test_training_arguments_coerces_yaml_boolean_no_save_strategy(tmp_path: Path) -> None:
    args = _coerce_training_arguments(
        {
            "output_dir": str(tmp_path / "out"),
            "save_strategy": False,
            "report_to": [],
        }
    )

    assert args.save_strategy == "no"
    assert args.logging_nan_inf_filter is False


def test_training_defaults_to_latest_checkpoint_in_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-10").mkdir(parents=True)
    (output_dir / "checkpoint-2").mkdir()
    args = _coerce_training_arguments(
        {
            "output_dir": str(output_dir),
            "save_strategy": "no",
            "report_to": [],
        }
    )

    assert _resolve_resume_from_checkpoint(args) == str(output_dir / "checkpoint-10")


def test_training_resume_from_checkpoint_can_be_disabled(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-10").mkdir(parents=True)
    args = _coerce_training_arguments(
        {
            "output_dir": str(output_dir),
            "resume_from_checkpoint": False,
            "save_strategy": "no",
            "report_to": [],
        }
    )

    assert _resolve_resume_from_checkpoint(args) is False


def test_configure_wandb_missing_section_is_noop(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    training_section = {"output_dir": "out"}

    configure_wandb(training_section, None)

    assert training_section == {"output_dir": "out"}
    assert "WANDB_PROJECT" not in os.environ


def test_configure_wandb_defined_section_sets_report_to_and_environment(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    training_section = {"output_dir": "out"}

    configure_wandb(
        training_section,
        {
            "project": "llava-anything",
            "entity": "research",
            "name": "stage1",
            "mode": "offline",
        },
    )

    assert training_section["report_to"] == ["wandb"]
    assert training_section["run_name"] == "stage1"
    assert os.environ["WANDB_PROJECT"] == "llava-anything"
    assert os.environ["WANDB_ENTITY"] == "research"
    assert os.environ["WANDB_MODE"] == "offline"


def test_configure_wandb_disabled_keeps_report_to_empty(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    training_section = {"output_dir": "out"}

    configure_wandb(training_section, {"enabled": False, "project": "ignored"})

    assert training_section["report_to"] == []
    assert "WANDB_PROJECT" not in os.environ
