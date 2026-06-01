from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


def save_tiny_text_component(
    path: Path,
    *,
    include_image_token: bool = True,
    model_vocab_size: int = 64,
) -> None:
    config = LlamaConfig(
        vocab_size=model_vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)

    vocab = {"<unk>": 0, "hello": 1}
    text_vocab_limit = model_vocab_size - 1 if include_image_token else model_vocab_size
    vocab.update({f"token_{idx}": idx for idx in range(2, text_vocab_limit)})
    if include_image_token:
        vocab["<image>"] = model_vocab_size - 1
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer_kwargs = {"tokenizer_object": raw, "unk_token": "<unk>"}
    if include_image_token:
        tokenizer_kwargs["additional_special_tokens"] = ["<image>"]
    tokenizer = PreTrainedTokenizerFast(**tokenizer_kwargs)
    tokenizer.save_pretrained(path)


def save_tiny_vision_component(path: Path) -> None:
    config = CLIPVisionConfig(
        image_size=8,
        patch_size=4,
        num_channels=3,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
    )
    model = CLIPVisionModel(config)
    model.save_pretrained(path)
    CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}).save_pretrained(path)


def write_tiny_model_yaml(path: Path, text_dir: Path, vision_dir: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "image_token": "<image>",
                    "image_token_index": 63,
                    "image_seq_length": 4,
                    "projector_type": "linear",
                    "vision_feature_layer": -1,
                    "vision_feature_select_strategy": "default",
                },
                "text_model": {"name_or_path": str(text_dir), "tokenizer": {}},
                "vision_model": {"name_or_path": str(vision_dir), "image_processor": {"patch_size": 4}},
            }
        )
    )


@pytest.fixture
def tiny_text_component_factory(tmp_path: Path) -> Callable[..., Path]:
    def factory(*, include_image_token: bool = True, model_vocab_size: int = 64) -> Path:
        suffix = "with-image-token" if include_image_token else "without-image-token"
        path = tmp_path / f"text-{suffix}-{model_vocab_size}"
        save_tiny_text_component(
            path,
            include_image_token=include_image_token,
            model_vocab_size=model_vocab_size,
        )
        return path

    return factory


@pytest.fixture
def tiny_text_component_dir(tiny_text_component_factory: Callable[..., Path]) -> Path:
    return tiny_text_component_factory()


@pytest.fixture
def tiny_vision_component_dir(tmp_path: Path) -> Path:
    path = tmp_path / "vision"
    save_tiny_vision_component(path)
    return path


@pytest.fixture
def tiny_model_yaml_path(tmp_path: Path, tiny_text_component_dir: Path, tiny_vision_component_dir: Path) -> Path:
    path = tmp_path / "model.yaml"
    write_tiny_model_yaml(path, tiny_text_component_dir, tiny_vision_component_dir)
    return path


@pytest.fixture
def tiny_model_yaml_factory(tmp_path: Path, tiny_vision_component_dir: Path) -> Callable[[Path], Path]:
    def factory(text_dir: Path) -> Path:
        path = tmp_path / f"model-{text_dir.name}.yaml"
        write_tiny_model_yaml(path, text_dir, tiny_vision_component_dir)
        return path

    return factory


@pytest.fixture
def tiny_full_model_dir(tmp_path: Path) -> Path:
    return tmp_path / "composed"


@pytest.fixture
def tiny_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color="white")
