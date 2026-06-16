from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_converter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "convert_llava_next_to_llava_anything.py"
    spec = importlib.util.spec_from_file_location("convert_llava_next_to_llava_anything", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_metadata_conversion_preserves_qwen2_backbone_and_infers_image_token(tmp_path: Path) -> None:
    converter = load_converter_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    write_json(
        source_dir / "config.json",
        {
            "_name_or_path": "Qwen/Qwen2-7B-Instruct",
            "architectures": ["LlavaQwenForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_aspect_ratio": "anyres",
            "image_grid_pinpoints": [[336, 672]],
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768,
            "mm_hidden_size": 1024,
            "mm_projector_type": "mlp2x_gelu",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
            "model_type": "qwen2",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "tokenizer_model_max_length": 8192,
            "tokenizer_padding_side": "right",
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 152064,
        },
    )
    write_json(
        source_dir / "tokenizer.json",
        {
            "model": {
                "vocab": {
                    "regular-token-at-999": 999,
                    "<|endoftext|>": 151643,
                    "<|im_start|>": 151644,
                    "<|im_end|>": 151645,
                }
            },
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": True},
                {"id": 151644, "content": "<|im_start|>", "special": True},
                {"id": 151645, "content": "<|im_end|>", "special": True},
            ],
        },
    )
    write_json(
        source_dir / "tokenizer_config.json",
        {
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>", "special": True},
                "151644": {"content": "<|im_start|>", "special": True},
                "151645": {"content": "<|im_end|>", "special": True},
            },
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "model_max_length": 8192,
            "padding_side": "right",
            "tokenizer_class": "Qwen2Tokenizer",
        },
    )
    write_json(
        source_dir / "special_tokens_map.json",
        {
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
    )

    args = argparse.Namespace(
        text_model_name_or_path=None,
        vision_model_name_or_path=None,
        vision_model_type=None,
        vision_image_size=None,
        vision_patch_size=None,
        vision_intermediate_size=None,
        image_token_id=None,
        prefer_backups=True,
    )

    converter.prepare_output_dir(output_dir, overwrite=False)
    converter.write_metadata(source_dir, output_dir, source_dir / "config.json", args)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))

    assert config["text_config"]["model_type"] == "qwen2"
    assert config["text_config"]["_name_or_path"] == "Qwen/Qwen2-7B-Instruct"
    assert config["text_config"]["vocab_size"] == 152064
    assert config["image_token_index"] == 151646
    processor = json.loads((output_dir / "processor_config.json").read_text(encoding="utf-8"))
    assert processor["image_processor"]["image_processor_type"] == "CLIPImageProcessor"
    assert tokenizer["model"]["vocab"]["regular-token-at-999"] == 999
    assert tokenizer["model"]["vocab"].get("<image>") is None
    assert any(item["content"] == "<image>" and item["id"] == 151646 for item in tokenizer["added_tokens"])


def test_qwen2_image_token_stays_out_of_base_vocab_to_avoid_special_id_collision(tmp_path: Path) -> None:
    converter = load_converter_module()
    tokenizer_path = tmp_path / "tokenizer.json"
    write_json(
        tokenizer_path,
        {
            "model": {
                "vocab": {
                    "regular-token-at-999": 999,
                }
            },
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": True},
                {"id": 151644, "content": "<|im_start|>", "special": True},
                {"id": 151645, "content": "<|im_end|>", "special": True},
            ],
        },
    )

    converter.rewrite_tokenizer_json(tokenizer_path, image_token_id=151646, tokenizer_class="Qwen2Tokenizer")

    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert tokenizer["model"]["vocab"].get("<image>") is None
    assert tokenizer["model"]["vocab"]["regular-token-at-999"] == 999
    assert tokenizer["added_tokens"][-1]["content"] == "<image>"
    assert tokenizer["added_tokens"][-1]["id"] == 151646
