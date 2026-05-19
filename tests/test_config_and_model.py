from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, CLIPVisionConfig, LlamaConfig

from llava_anything import LlavaAnythingConfig, LlavaAnythingForConditionalGeneration


def tiny_config() -> LlavaAnythingConfig:
    text_config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    vision_config = CLIPVisionConfig(
        image_size=8,
        patch_size=4,
        num_channels=3,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
    )
    return LlavaAnythingConfig.from_text_vision_configs(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=63,
        image_seq_length=4,
        projector_type="linear",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    )


def test_config_round_trips_nested_configs() -> None:
    config = tiny_config()
    restored = LlavaAnythingConfig.from_dict(config.to_dict())

    assert restored.model_type == "llava_anything"
    assert restored.text_config.model_type == "llama"
    assert restored.vision_config.model_type == "clip_vision_model"
    assert restored.image_seq_length == 4


def test_forward_replaces_image_placeholders() -> None:
    torch.manual_seed(0)
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    input_ids = torch.tensor([[1, 63, 63, 63, 63, 2]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 8, 8)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

    assert outputs.logits.shape == (1, 6, 64)
    assert outputs.image_hidden_states.shape == (1, 4, 16)


def test_forward_validates_image_token_count() -> None:
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    input_ids = torch.tensor([[1, 63, 63, 2]])
    pixel_values = torch.randn(1, 3, 8, 8)

    try:
        model(input_ids=input_ids, pixel_values=pixel_values)
    except ValueError as exc:
        assert "Image features and image tokens do not match" in str(exc)
    else:
        raise AssertionError("Expected image token mismatch to raise ValueError")


def test_generate_accepts_image_inputs() -> None:
    torch.manual_seed(0)
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    input_ids = torch.tensor([[1, 63, 63, 63, 63, 2]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 8, 8)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=1,
        do_sample=False,
    )

    assert output.shape == (1, 7)


def test_auto_model_for_image_text_to_text_reloads_saved_weights(tmp_path) -> None:
    torch.manual_seed(0)
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    model.save_pretrained(tmp_path)

    reloaded = AutoModelForImageTextToText.from_pretrained(tmp_path)

    assert isinstance(reloaded, LlavaAnythingForConditionalGeneration)
    assert reloaded.config.model_type == "llava_anything"
    assert reloaded.get_input_embeddings().num_embeddings == model.get_input_embeddings().num_embeddings


def test_wrapper_dtype_tracks_language_input_embeddings() -> None:
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    model.language_model.to(dtype=torch.bfloat16)

    assert model.dtype == torch.bfloat16
