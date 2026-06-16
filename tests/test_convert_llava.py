from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_converter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "convert_llava_to_llava_anything.py"
    spec = importlib.util.spec_from_file_location("convert_llava_to_llava_anything", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_old_palo_metadata_maps_to_llama_and_clip_defaults(tmp_path: Path) -> None:
    converter = load_converter_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = {
        "_name_or_path": "lmsys/vicuna-7b-v1.5",
        "architectures": ["PaloForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "image_aspect_ratio": "pad",
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "mm_hidden_size": 1024,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        "model_type": "palo",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "tokenizer_model_max_length": 2048,
        "tokenizer_padding_side": "right",
        "torch_dtype": "float16",
        "use_cache": True,
        "vocab_size": 32000,
    }
    write_json(source_dir / "config.json", source_config)
    write_json(
        source_dir / "pytorch_model.bin.index.json",
        {
            "weight_map": {
                "model.vision_tower.vision_tower.vision_model.embeddings.class_embedding": "pytorch_model.bin",
                "model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": "pytorch_model.bin",
                "model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj.weight": "pytorch_model.bin",
            }
        },
    )

    image_token_id = converter.resolve_image_token_id(source_config, requested_image_token_id=None)
    config = converter.build_llava_anything_config(
        llava_config=source_config,
        source_dir=source_dir,
        text_model_name_or_path=None,
        vision_model_name_or_path=None,
        vision_model_type=None,
        image_token_id=image_token_id,
        vocab_size=32001,
    )
    processor_config = converter.build_processor_config(
        llava_config=source_config,
        vision_image_size=config.vision_config.image_size,
        vision_patch_size=config.vision_config.patch_size,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        num_additional_image_tokens=config.num_additional_image_tokens,
    )

    assert image_token_id == 32000
    assert config.text_config.model_type == "llama"
    assert config.text_config.architectures == ["LlamaForCausalLM"]
    assert config.text_config.vocab_size == 32001
    assert config.image_token_index == 32000
    assert config.vision_config.model_type == "clip_vision_model"
    assert config.vision_config.num_hidden_layers == 24
    assert config.vision_feature_select_strategy == "default"
    assert config.num_additional_image_tokens == 1
    assert processor_config["image_processor"]["image_processor_type"] == "CLIPImageProcessor"


def test_sanitize_generation_config_drops_sampling_flags_when_not_sampling() -> None:
    converter = load_converter_module()

    sanitized = converter.sanitize_generation_config(
        {
            "do_sample": False,
            "temperature": 0.9,
            "top_p": 0.6,
            "max_new_tokens": 64,
        }
    )

    assert sanitized == {"do_sample": False, "max_new_tokens": 64}


def test_old_llava_without_embedded_vision_weights_assumes_clip_class_token(tmp_path: Path) -> None:
    converter = load_converter_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = {
        "_name_or_path": "liuhaotian/llava-v1.5-7b",
        "hidden_size": 4096,
        "image_aspect_ratio": "pad",
        "intermediate_size": 11008,
        "mm_hidden_size": 1024,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        "model_type": "llava",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
    }
    write_json(
        source_dir / "pytorch_model.bin.index.json",
        {
            "weight_map": {
                "model.embed_tokens.weight": "pytorch_model.bin",
                "model.mm_projector.0.weight": "pytorch_model.bin",
            }
        },
    )

    config = converter.build_llava_anything_config(
        llava_config=source_config,
        source_dir=source_dir,
        text_model_name_or_path=None,
        vision_model_name_or_path=None,
        vision_model_type=None,
        image_token_id=32000,
        vocab_size=32001,
    )

    assert config.vision_feature_select_strategy == "default"
    assert config.num_additional_image_tokens == 1


def test_special_tokens_map_includes_image_token() -> None:
    converter = load_converter_module()

    class Tokenizer:
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        pad_token = "<unk>"
        additional_special_tokens = ["<image>"]

    assert converter.special_tokens_map_from_tokenizer(Tokenizer()) == {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
        "additional_special_tokens": ["<image>"],
    }


def test_missing_vision_weights_are_loaded_from_configured_vision_tower(tmp_path: Path, monkeypatch) -> None:
    converter = load_converter_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_json(
        source_dir / "pytorch_model.bin.index.json",
        {
            "weight_map": {
                "model.embed_tokens.weight": "pytorch_model.bin",
                "model.mm_projector.0.weight": "pytorch_model.bin",
            }
        },
    )
    model = SimpleNamespace(vision_tower="initialized")
    config = SimpleNamespace(
        vision_model_name_or_path="/models/openai/clip-vit-large-patch14-336",
        vision_trust_remote_code=False,
    )
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(vision_model="loaded-clip-vision")

    monkeypatch.setattr(converter, "AutoModel", FakeAutoModel)

    converter.replace_missing_vision_tower(model, source_dir, config, torch_dtype="float16")

    assert model.vision_tower == "loaded-clip-vision"
    assert calls == [
        (
            "/models/openai/clip-vit-large-patch14-336",
            {"torch_dtype": "float16", "trust_remote_code": False},
        )
    ]
