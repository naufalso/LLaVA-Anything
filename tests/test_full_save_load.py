from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor

from llava_anything import LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor
from llava_anything.builder import save_from_yaml


def test_builder_saves_full_composed_model_reloadable_by_auto_classes(
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)

    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)

    assert isinstance(processor, LlavaAnythingProcessor)
    assert isinstance(model, LlavaAnythingForConditionalGeneration)
    assert model.get_input_embeddings().num_embeddings == len(processor.tokenizer)


def test_full_composition_saves_expected_artifact_files(
    tiny_model_yaml_path: Path,
    tiny_full_model_dir: Path,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)

    saved_files = {path.name for path in tiny_full_model_dir.iterdir() if path.is_file()}
    assert "config.json" in saved_files
    assert "tokenizer.json" in saved_files
    assert "tokenizer_config.json" in saved_files
    assert "processor_config.json" in saved_files
    processor_config = json.loads((tiny_full_model_dir / "processor_config.json").read_text())
    assert processor_config["image_processor"]["image_processor_type"] == "CLIPImageProcessor"
    assert {"model.safetensors", "pytorch_model.bin"} & saved_files


def test_full_composition_resizes_embeddings_for_added_image_token_before_saving(
    tiny_text_component_factory: Callable[..., Path],
    tiny_model_yaml_factory: Callable[[Path], Path],
    tiny_full_model_dir: Path,
) -> None:
    text_dir = tiny_text_component_factory(include_image_token=False, model_vocab_size=63)
    yaml_path = tiny_model_yaml_factory(text_dir)

    save_from_yaml(yaml_path, tiny_full_model_dir, load_pretrained_components=True)

    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)

    assert processor.tokenizer.convert_tokens_to_ids("<image>") == 63
    assert model.get_input_embeddings().num_embeddings == len(processor.tokenizer) == 64
    assert model.config.vocab_size == 64
    assert model.config.text_config.vocab_size == 64
