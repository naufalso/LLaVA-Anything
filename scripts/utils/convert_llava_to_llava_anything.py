#!/usr/bin/env python
"""Convert an older LLaVA checkpoint directory to LLaVA-Anything format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

import llava_anything  # noqa: F401 - registers Auto classes
from llava_anything.configuration_llava_anything import LlavaAnythingConfig
from llava_anything.modeling_llava_anything import LlavaAnythingForConditionalGeneration


IMAGE_TOKEN = "<image>"
DEFAULT_TEXT_MODEL = "lmsys/vicuna-7b-v1.5"
CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
SAMPLING_ONLY_GENERATION_KEYS = {
    "temperature",
    "top_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sanitize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(config)
    if not bool(sanitized.get("do_sample", False)):
        for key in SAMPLING_ONLY_GENERATION_KEYS:
            sanitized.pop(key, None)
    return sanitized


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def dtype_from_arg(value: str) -> torch.dtype | str | None:
    if value == "auto":
        return "auto"
    if value.lower() in {"none", "null"}:
        return None
    dtype = getattr(torch, value, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}")


def weight_map(source_dir: Path) -> dict[str, str]:
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = source_dir / filename
        if path.exists():
            mapping = load_json(path).get("weight_map", {})
            if isinstance(mapping, dict):
                return {str(key): str(value) for key, value in mapping.items()}
    return {}


def infer_vision_num_hidden_layers(source_dir: Path, default: int = 24) -> int:
    mapping = weight_map(source_dir)
    prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
    layer_indexes = []
    for key in mapping:
        if key.startswith(prefix):
            layer_index = key.removeprefix(prefix).split(".", 1)[0]
            if layer_index.isdigit():
                layer_indexes.append(int(layer_index))
    return max(layer_indexes) + 1 if layer_indexes else default


def source_has_vision_weights(source_dir: Path) -> bool:
    mapping = weight_map(source_dir)
    return any(
        key.startswith("model.vision_tower.vision_tower.")
        or key.startswith("model.vision_tower.vision_tower.vision_model.")
        for key in mapping
    )


def infer_vision_has_class_embedding(source_dir: Path, llava_config: dict[str, Any] | None = None) -> bool:
    mapping = weight_map(source_dir)
    if "model.vision_tower.vision_tower.vision_model.embeddings.class_embedding" in mapping:
        return True
    if any(key.startswith("model.vision_tower.vision_tower.vision_model.") for key in mapping):
        return False
    if llava_config is not None and infer_vision_model_type(llava_config, fallback=None) == "clip_vision_model":
        return True
    return False


def infer_vision_model_type(llava_config: dict[str, Any], fallback: str | None) -> str:
    if fallback:
        return fallback
    tower = str(llava_config.get("mm_vision_tower", "")).lower()
    if "siglip" in tower:
        return "siglip_vision_model"
    return "clip_vision_model"


def infer_vision_patch_size(llava_config: dict[str, Any]) -> int:
    tower = str(llava_config.get("mm_vision_tower", "")).lower()
    if "patch14" in tower or "patch-14" in tower:
        return 14
    return 14


def infer_vision_image_size(llava_config: dict[str, Any]) -> int:
    tower = str(llava_config.get("mm_vision_tower", "")).lower()
    if "336" in tower:
        return 336
    if "384" in tower:
        return 384
    return 336


def infer_vision_intermediate_size(hidden_size: int) -> int:
    if hidden_size == 1024:
        return 4096
    if hidden_size == 1152:
        return 4304
    return hidden_size * 4


def infer_text_model_name_or_path(llava_config: dict[str, Any], fallback: str | None) -> str:
    if fallback:
        return fallback
    name_or_path = llava_config.get("_name_or_path")
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path
    return DEFAULT_TEXT_MODEL


def infer_text_model_type(llava_config: dict[str, Any]) -> str:
    model_type = str(llava_config.get("model_type", "")).lower()
    name_or_path = str(llava_config.get("_name_or_path", "")).lower()
    if model_type in {"palo", "llava", "llava_llama"} or "vicuna" in name_or_path:
        return "llama"
    if model_type:
        return model_type
    return "llama"


def infer_text_architectures(text_model_type: str) -> list[str] | None:
    if text_model_type == "llama":
        return ["LlamaForCausalLM"]
    if text_model_type == "qwen2":
        return ["Qwen2ForCausalLM"]
    return None


def resolve_image_token_id(llava_config: dict[str, Any], requested_image_token_id: int | None) -> int:
    if requested_image_token_id is not None:
        return int(requested_image_token_id)
    existing = llava_config.get("image_token_index")
    if existing is not None:
        return int(existing)
    return int(llava_config["vocab_size"])


def build_text_config(
    llava_config: dict[str, Any],
    text_model_name_or_path: str | None,
    vocab_size: int,
) -> dict[str, Any]:
    text_keys = [
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "pretraining_tp",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "tie_word_embeddings",
        "torch_dtype",
        "use_cache",
    ]
    text_model_type = infer_text_model_type(llava_config)
    text_config = {key: llava_config[key] for key in text_keys if key in llava_config}
    text_config.update(
        {
            "_name_or_path": infer_text_model_name_or_path(llava_config, text_model_name_or_path),
            "model_type": text_model_type,
            "vocab_size": int(vocab_size),
        }
    )
    architectures = infer_text_architectures(text_model_type)
    if architectures is not None:
        text_config["architectures"] = architectures
    return text_config


def build_llava_anything_config(
    llava_config: dict[str, Any],
    source_dir: Path,
    text_model_name_or_path: str | None,
    vision_model_name_or_path: str | None,
    vision_model_type: str | None,
    image_token_id: int,
    vocab_size: int,
) -> LlavaAnythingConfig:
    vision_hidden_size = int(llava_config.get("mm_hidden_size", 1024))
    vision_patch_size = infer_vision_patch_size(llava_config)
    vision_image_size = infer_vision_image_size(llava_config)
    has_class_embedding = infer_vision_has_class_embedding(source_dir, llava_config)
    vision_feature_select_strategy = "default" if has_class_embedding else "full"
    num_additional_image_tokens = 1 if has_class_embedding else 0
    resolved_text_model_name_or_path = infer_text_model_name_or_path(llava_config, text_model_name_or_path)

    vision_config = {
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": vision_hidden_size,
        "image_size": vision_image_size,
        "intermediate_size": infer_vision_intermediate_size(vision_hidden_size),
        "layer_norm_eps": 1e-5,
        "model_type": infer_vision_model_type(llava_config, vision_model_type),
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": infer_vision_num_hidden_layers(source_dir),
        "patch_size": vision_patch_size,
        "vision_use_head": False,
    }

    config = LlavaAnythingConfig(
        text_config=build_text_config(llava_config, resolved_text_model_name_or_path, vocab_size),
        vision_config=vision_config,
        image_token=IMAGE_TOKEN,
        image_token_index=int(image_token_id),
        projector_type=llava_config.get("mm_projector_type", "mlp2x_gelu"),
        projector_hidden_act="gelu",
        vision_feature_layer=llava_config.get("mm_vision_select_layer", -2),
        vision_feature_select_strategy=vision_feature_select_strategy,
        image_seq_length=(vision_image_size // vision_patch_size) ** 2,
        num_additional_image_tokens=num_additional_image_tokens,
        image_mode="anyres" if llava_config.get("image_aspect_ratio") == "anyres" else "fixed",
        image_grid_pinpoints=llava_config.get("image_grid_pinpoints"),
        text_model_name_or_path=resolved_text_model_name_or_path,
        vision_model_name_or_path=vision_model_name_or_path or llava_config.get("mm_vision_tower"),
        trust_remote_code=False,
        text_trust_remote_code=False,
        vision_trust_remote_code=False,
        dtype=llava_config.get("dtype", llava_config.get("torch_dtype")),
        hidden_size=llava_config.get("hidden_size"),
        vocab_size=int(vocab_size),
        use_cache=llava_config.get("use_cache", True),
        legacy_llava_next_checkpoint=True,
    )
    config.architectures = ["LlavaAnythingForConditionalGeneration"]
    return config


def build_processor_config(
    llava_config: dict[str, Any],
    vision_image_size: int,
    vision_patch_size: int,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> dict[str, Any]:
    return {
        "image_grid_pinpoints": llava_config.get("image_grid_pinpoints"),
        "image_mode": "anyres" if llava_config.get("image_aspect_ratio") == "anyres" else "fixed",
        "image_processor": {
            "crop_size": {"height": int(vision_image_size), "width": int(vision_image_size)},
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": CLIP_IMAGE_MEAN,
            "image_processor_type": "CLIPImageProcessor",
            "image_std": CLIP_IMAGE_STD,
            "resample": 3,
            "rescale_factor": 1 / 255,
            "size": {"shortest_edge": int(vision_image_size)},
        },
        "image_seq_length": (int(vision_image_size) // int(vision_patch_size)) ** 2,
        "image_token": IMAGE_TOKEN,
        "num_additional_image_tokens": int(num_additional_image_tokens),
        "patch_size": int(vision_patch_size),
        "processor_class": "LlavaAnythingProcessor",
        "vision_feature_select_strategy": vision_feature_select_strategy,
    }


def write_tokenizer(source_dir: Path, output_dir: Path, llava_config: dict[str, Any], image_token_id: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    additional_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if IMAGE_TOKEN not in additional_tokens:
        additional_tokens.append(IMAGE_TOKEN)
    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
    actual_image_token_id = int(tokenizer.convert_tokens_to_ids(IMAGE_TOKEN))
    if actual_image_token_id != int(image_token_id):
        raise ValueError(
            f"Tokenizer assigned {IMAGE_TOKEN} id {actual_image_token_id}, expected {image_token_id}. "
            "Pass --image-token-id matching the tokenizer append position or use the default."
        )
    tokenizer.model_max_length = int(llava_config.get("tokenizer_model_max_length", tokenizer.model_max_length))
    tokenizer.padding_side = llava_config.get("tokenizer_padding_side", tokenizer.padding_side)
    tokenizer.save_pretrained(output_dir)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = load_json(tokenizer_config_path)
    tokenizer_config["image_token"] = IMAGE_TOKEN
    tokenizer_config["model_max_length"] = int(llava_config.get("tokenizer_model_max_length", tokenizer_config.get("model_max_length", 4096)))
    tokenizer_config["padding_side"] = llava_config.get("tokenizer_padding_side", tokenizer_config.get("padding_side", "right"))
    write_json(tokenizer_config_path, tokenizer_config)
    write_json(output_dir / "special_tokens_map.json", special_tokens_map_from_tokenizer(tokenizer))


def special_tokens_map_from_tokenizer(tokenizer: Any) -> dict[str, Any]:
    special_tokens: dict[str, Any] = {}
    for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = getattr(tokenizer, key, None)
        if value is not None:
            special_tokens[key] = value
    additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if additional:
        special_tokens["additional_special_tokens"] = additional
    return special_tokens


def write_metadata(
    source_dir: Path,
    output_dir: Path,
    llava_config: dict[str, Any],
    config: LlavaAnythingConfig,
    image_token_id: int,
) -> None:
    config.save_pretrained(output_dir)
    write_json(
        output_dir / "processor_config.json",
        build_processor_config(
            llava_config,
            config.vision_config.image_size,
            config.vision_config.patch_size,
            config.vision_feature_select_strategy,
            config.num_additional_image_tokens,
        ),
    )
    write_tokenizer(source_dir, output_dir, llava_config, image_token_id)
    generation_config = source_dir / "generation_config.json"
    if generation_config.exists():
        write_json(output_dir / "generation_config.json", sanitize_generation_config(load_json(generation_config)))


def resize_token_embeddings_if_needed(
    model: LlavaAnythingForConditionalGeneration,
    final_vocab_size: int,
) -> None:
    current_vocab_size = int(model.get_input_embeddings().num_embeddings)
    if final_vocab_size <= current_vocab_size:
        return
    try:
        model.resize_token_embeddings(final_vocab_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(final_vocab_size)
    model.config.text_config.vocab_size = final_vocab_size
    model.config.vocab_size = final_vocab_size
    if hasattr(model.language_model, "config"):
        model.language_model.config.vocab_size = final_vocab_size


def replace_missing_vision_tower(
    model: LlavaAnythingForConditionalGeneration,
    source_dir: Path,
    config: LlavaAnythingConfig,
    torch_dtype: str | None,
) -> None:
    if source_has_vision_weights(source_dir):
        return
    if not config.vision_model_name_or_path:
        raise ValueError("Source checkpoint has no vision tower weights and no vision_model_name_or_path is configured.")

    loaded_vision_tower = AutoModel.from_pretrained(
        config.vision_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(config.vision_trust_remote_code),
    )
    model.vision_tower = getattr(loaded_vision_tower, "vision_model", loaded_vision_tower)


def convert_model_weights(
    source_dir: Path,
    output_dir: Path,
    load_config: LlavaAnythingConfig,
    final_vocab_size: int,
    args: argparse.Namespace,
) -> LlavaAnythingForConditionalGeneration:
    model_kwargs: dict[str, Any] = {
        "config": load_config,
        "key_mapping": dict(LlavaAnythingForConditionalGeneration._legacy_llava_next_key_mapping),
    }
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = dtype_from_arg(args.torch_dtype)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = LlavaAnythingForConditionalGeneration.from_pretrained(source_dir, **model_kwargs)
    resize_token_embeddings_if_needed(model, final_vocab_size)
    replace_missing_vision_tower(model, source_dir, model.config, args.torch_dtype)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config = GenerationConfig.from_dict(sanitize_generation_config(model.generation_config.to_dict()))
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        save_original_format=False,
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Older LLaVA checkpoint directory.")
    parser.add_argument("output_dir", type=Path, help="Directory for the converted LLaVA-Anything checkpoint.")
    parser.add_argument("--text-model-name-or-path")
    parser.add_argument("--vision-model-name-or-path")
    parser.add_argument("--vision-model-type")
    parser.add_argument("--image-token-id", type=int, help="Defaults to the original tokenizer vocab size.")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    llava_config = load_json(source_dir / "config.json")
    source_vocab_size = int(llava_config["vocab_size"])
    image_token_id = resolve_image_token_id(llava_config, args.image_token_id)
    final_vocab_size = max(source_vocab_size, int(image_token_id) + 1)

    prepare_output_dir(output_dir, args.overwrite)
    if args.metadata_only:
        config = build_llava_anything_config(
            llava_config=llava_config,
            source_dir=source_dir,
            text_model_name_or_path=args.text_model_name_or_path,
            vision_model_name_or_path=args.vision_model_name_or_path,
            vision_model_type=args.vision_model_type,
            image_token_id=image_token_id,
            vocab_size=final_vocab_size,
        )
        write_metadata(source_dir, output_dir, llava_config, config, image_token_id)
    else:
        load_config = build_llava_anything_config(
            llava_config=llava_config,
            source_dir=source_dir,
            text_model_name_or_path=args.text_model_name_or_path,
            vision_model_name_or_path=args.vision_model_name_or_path,
            vision_model_type=args.vision_model_type,
            image_token_id=image_token_id,
            vocab_size=source_vocab_size,
        )
        model = convert_model_weights(source_dir, output_dir, load_config, final_vocab_size, args)
        write_metadata(source_dir, output_dir, llava_config, model.config, image_token_id)
    print(f"Saved LLaVA-Anything conversion to {output_dir}")


if __name__ == "__main__":
    main()
