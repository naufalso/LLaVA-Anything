from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

from llava_anything.builder import save_from_yaml
from llava_anything.training import (
    IGNORE_INDEX,
    LlavaPretrainDataCollator,
    LlavaPretrainDataset,
    apply_trainable_modules,
    configure_wandb,
    log_preview_samples,
    run_pretraining_from_yaml,
    _coerce_training_arguments,
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
