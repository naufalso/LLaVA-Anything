from __future__ import annotations

import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llava_anything import LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor
from llava_anything.builder import save_from_yaml


def _save_tiny_text_component(path, *, include_image_token: bool = True, model_vocab_size: int = 64) -> None:
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
    vocab.update({f"token_{idx}": idx for idx in range(2, model_vocab_size)})
    if include_image_token:
        vocab["<image>"] = len(vocab)
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer_kwargs = {"tokenizer_object": raw, "unk_token": "<unk>"}
    if include_image_token:
        tokenizer_kwargs["additional_special_tokens"] = ["<image>"]
    tokenizer = PreTrainedTokenizerFast(**tokenizer_kwargs)
    tokenizer.save_pretrained(path)


def _save_tiny_vision_component(path) -> None:
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


def test_builder_saves_full_composed_model_reloadable_by_auto_classes(tmp_path) -> None:
    text_dir = tmp_path / "text"
    vision_dir = tmp_path / "vision"
    output_dir = tmp_path / "composed"
    _save_tiny_text_component(text_dir)
    _save_tiny_vision_component(vision_dir)
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
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
                "vision_model": {
                    "name_or_path": str(vision_dir),
                    "image_processor": {"patch_size": 4},
                },
            }
        )
    )

    save_from_yaml(yaml_path, output_dir, load_pretrained_components=True)

    processor = AutoProcessor.from_pretrained(output_dir)
    model = AutoModelForImageTextToText.from_pretrained(output_dir)

    assert isinstance(processor, LlavaAnythingProcessor)
    assert isinstance(model, LlavaAnythingForConditionalGeneration)
    assert model.get_input_embeddings().num_embeddings == len(processor.tokenizer)



def test_full_composition_saves_expected_artifact_files(tmp_path) -> None:
    text_dir = tmp_path / "text"
    vision_dir = tmp_path / "vision"
    output_dir = tmp_path / "composed"
    _save_tiny_text_component(text_dir)
    _save_tiny_vision_component(vision_dir)
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
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

    save_from_yaml(yaml_path, output_dir, load_pretrained_components=True)

    saved_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert "config.json" in saved_files
    assert "tokenizer.json" in saved_files
    assert "tokenizer_config.json" in saved_files
    assert "preprocessor_config.json" in saved_files
    assert "processor_config.json" in saved_files
    assert {"model.safetensors", "pytorch_model.bin"} & saved_files


def test_full_composition_resizes_embeddings_for_added_image_token_before_saving(tmp_path) -> None:
    text_dir = tmp_path / "text"
    vision_dir = tmp_path / "vision"
    output_dir = tmp_path / "composed"
    _save_tiny_text_component(text_dir, include_image_token=False, model_vocab_size=63)
    _save_tiny_vision_component(vision_dir)
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
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

    save_from_yaml(yaml_path, output_dir, load_pretrained_components=True)

    processor = AutoProcessor.from_pretrained(output_dir)
    model = AutoModelForImageTextToText.from_pretrained(output_dir)

    assert processor.tokenizer.convert_tokens_to_ids("<image>") == 63
    assert model.get_input_embeddings().num_embeddings == len(processor.tokenizer) == 64
    assert model.config.vocab_size == 64
    assert model.config.text_config.vocab_size == 64
