#!/usr/bin/env python
"""Convert a LLaVA-NeXT checkpoint directory to LLaVA-Anything format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

import llava_anything  # noqa: F401 - registers Auto classes
from llava_anything.configuration_llava_anything import LlavaAnythingConfig
from llava_anything.modeling_llava_anything import LlavaAnythingForConditionalGeneration


DEFAULT_TEXT_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_IMAGE_TOKEN_ID = 999


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def existing_source_file(source_dir: Path, filename: str, prefer_backup: bool = True) -> Path:
    backup = source_dir / f"{filename}.llava-next.bak"
    if prefer_backup and backup.exists():
        return backup
    path = source_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")
    return path


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_if_exists(source_dir: Path, output_dir: Path, filename: str, prefer_backup: bool = False) -> None:
    candidates = []
    if prefer_backup:
        candidates.append(source_dir / f"{filename}.llava-next.bak")
    candidates.append(source_dir / filename)
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, output_dir / filename)
            return


def infer_vision_num_hidden_layers(source_dir: Path, default: int = 26) -> int:
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return default
    weight_map = load_json(index_path).get("weight_map", {})
    layer_indexes = []
    prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
    for key in weight_map:
        if key.startswith(prefix):
            remainder = key.removeprefix(prefix)
            layer_index = remainder.split(".", 1)[0]
            if layer_index.isdigit():
                layer_indexes.append(int(layer_index))
    return max(layer_indexes) + 1 if layer_indexes else default


def legacy_vision_key(source_key: str) -> str:
    return f"model.vision_tower.vision_tower.vision_model.{source_key}"


def tensor_shape_from_index(source_dir: Path, key: str) -> list[int] | None:
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    weight_map = load_json(index_path).get("weight_map", {})
    shard_name = weight_map.get(key)
    if not shard_name:
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    with safe_open(source_dir / shard_name, framework="pt", device="cpu") as handle:
        return [int(dim) for dim in handle.get_slice(key).get_shape()]


def infer_vision_patch_size(source_dir: Path, fallback: int) -> int:
    shape = tensor_shape_from_index(source_dir, legacy_vision_key("embeddings.patch_embedding.weight"))
    if shape is not None and len(shape) >= 4:
        return int(shape[-1])
    return int(fallback)


def infer_vision_image_size(source_dir: Path, patch_size: int, fallback: int) -> int:
    position_shape = tensor_shape_from_index(source_dir, legacy_vision_key("embeddings.position_embedding.weight"))
    has_class_embedding = infer_vision_has_class_embedding(source_dir)
    if position_shape is not None and position_shape:
        num_positions = int(position_shape[0]) - (1 if has_class_embedding else 0)
        grid_size = int(num_positions**0.5)
        if grid_size * grid_size == num_positions:
            return grid_size * int(patch_size)
    return int(fallback)


def infer_vision_has_class_embedding(source_dir: Path) -> bool:
    return tensor_shape_from_index(source_dir, legacy_vision_key("embeddings.class_embedding")) is not None


def infer_vision_intermediate_size(source_dir: Path, fallback: int) -> int:
    shape = tensor_shape_from_index(source_dir, legacy_vision_key("encoder.layers.0.mlp.fc1.weight"))
    if shape is not None and shape:
        return int(shape[0])
    return int(fallback)


def infer_vision_hidden_size(source_dir: Path, fallback: int) -> int:
    shape = tensor_shape_from_index(source_dir, legacy_vision_key("embeddings.patch_embedding.weight"))
    if shape is not None and shape:
        return int(shape[0])
    return int(fallback)


def infer_vision_use_head(source_dir: Path) -> bool:
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    weight_map = load_json(index_path).get("weight_map", {})
    return any(".head." in key for key in weight_map)


def infer_vision_model_type(llava_next_config: dict[str, Any], fallback: str | None) -> str:
    if fallback:
        return fallback
    tower = str(llava_next_config.get("mm_vision_tower", "")).lower()
    if "clip" in tower:
        return "clip_vision_model"
    if "siglip2" in tower:
        return "siglip_vision_model"
    if "siglip" in tower:
        return "siglip_vision_model"
    return "clip_vision_model"


def build_llava_anything_config(
    llava_next_config: dict[str, Any],
    source_dir: Path,
    text_model_name_or_path: str,
    vision_model_name_or_path: str | None,
    vision_model_type: str | None,
    vision_image_size: int | None,
    vision_patch_size: int | None,
    vision_intermediate_size: int | None,
    image_token_id: int,
) -> LlavaAnythingConfig:
    text_keys = [
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "hidden_act",
        "hidden_dropout",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "mlp_bias",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "post_norm",
        "qk_norm",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "tie_word_embeddings",
        "use_cache",
    ]
    text_config = {key: llava_next_config[key] for key in text_keys if key in llava_next_config}
    text_config.update(
        {
            "_name_or_path": text_model_name_or_path,
            "architectures": ["ApertusForCausalLM"],
            "model_type": "apertus",
            "vocab_size": int(llava_next_config["vocab_size"]),
        }
    )

    inferred_patch_size = infer_vision_patch_size(source_dir, vision_patch_size or 14)
    inferred_image_size = infer_vision_image_size(source_dir, inferred_patch_size, vision_image_size or 336)
    inferred_hidden_size = infer_vision_hidden_size(source_dir, int(llava_next_config.get("mm_hidden_size", 1024)))
    inferred_intermediate_size = infer_vision_intermediate_size(
        source_dir,
        vision_intermediate_size or (4304 if inferred_hidden_size == 1152 else 4096),
    )
    has_class_embedding = infer_vision_has_class_embedding(source_dir)
    vision_feature_select_strategy = "default" if has_class_embedding else "full"
    num_additional_image_tokens = 1 if has_class_embedding else 0

    vision_config = {
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": inferred_hidden_size,
        "image_size": inferred_image_size,
        "intermediate_size": inferred_intermediate_size,
        "layer_norm_eps": 1e-6,
        "model_type": infer_vision_model_type(llava_next_config, vision_model_type),
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": infer_vision_num_hidden_layers(source_dir),
        "patch_size": inferred_patch_size,
        "vision_use_head": infer_vision_use_head(source_dir),
    }

    config = LlavaAnythingConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token="<image>",
        image_token_index=int(image_token_id),
        projector_type=llava_next_config.get("mm_projector_type", "mlp2x_gelu"),
        projector_hidden_act="gelu",
        vision_feature_layer=llava_next_config.get("mm_vision_select_layer", -2),
        vision_feature_select_strategy=vision_feature_select_strategy,
        image_seq_length=(inferred_image_size // inferred_patch_size) ** 2,
        num_additional_image_tokens=num_additional_image_tokens,
        image_mode="anyres" if llava_next_config.get("image_aspect_ratio") == "anyres" else "fixed",
        image_grid_pinpoints=llava_next_config.get("image_grid_pinpoints"),
        text_model_name_or_path=text_model_name_or_path,
        vision_model_name_or_path=vision_model_name_or_path or llava_next_config.get("mm_vision_tower"),
        trust_remote_code=False,
        text_trust_remote_code=False,
        vision_trust_remote_code=False,
        dtype=llava_next_config.get("dtype"),
        hidden_size=llava_next_config.get("hidden_size"),
        vocab_size=int(llava_next_config["vocab_size"]),
        use_cache=llava_next_config.get("use_cache", True),
        legacy_llava_next_checkpoint=True,
    )
    config.architectures = ["LlavaAnythingForConditionalGeneration"]
    return config


def build_processor_config(
    llava_next_config: dict[str, Any],
    vision_image_size: int,
    vision_patch_size: int,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> dict[str, Any]:
    return {
        "image_grid_pinpoints": llava_next_config.get("image_grid_pinpoints"),
        "image_mode": "anyres" if llava_next_config.get("image_aspect_ratio") == "anyres" else "fixed",
        "image_processor": {
            "do_convert_rgb": None,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "SiglipImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "resample": 2,
            "rescale_factor": 1 / 255,
            "size": {"height": int(vision_image_size), "width": int(vision_image_size)},
        },
        "image_seq_length": (int(vision_image_size) // int(vision_patch_size)) ** 2,
        "image_token": "<image>",
        "num_additional_image_tokens": int(num_additional_image_tokens),
        "patch_size": int(vision_patch_size),
        "processor_class": "LlavaAnythingProcessor",
        "vision_feature_select_strategy": vision_feature_select_strategy,
    }


def rewrite_tokenizer_json(path: Path, image_token_id: int) -> None:
    tokenizer = load_json(path)
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"Expected tokenizer model vocab mapping in {path}")

    image_token_id = int(image_token_id)
    for token, token_id in list(vocab.items()):
        if int(token_id) == image_token_id:
            del vocab[token]
            break
    stale_image_id = vocab.get("<image>")
    if stale_image_id is not None and int(stale_image_id) != image_token_id:
        del vocab["<image>"]
    vocab["<image>"] = image_token_id

    added_tokens = tokenizer.setdefault("added_tokens", [])
    added_tokens = [
        item
        for item in added_tokens
        if item.get("content") != "<image>" and int(item.get("id", -1)) != image_token_id
    ]
    added_tokens.append(
        {
            "id": image_token_id,
            "content": "<image>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    tokenizer["added_tokens"] = sorted(added_tokens, key=lambda item: int(item.get("id", 0)))
    path.write_text(json.dumps(tokenizer, indent=None, separators=(",", ":")) + "\n", encoding="utf-8")


def rewrite_tokenizer_config(path: Path, image_token_id: int, llava_next_config: dict[str, Any]) -> None:
    config = load_json(path)
    decoder = config.setdefault("added_tokens_decoder", {})
    decoder[str(image_token_id)] = {
        "content": "<image>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    for token_id, item in list(decoder.items()):
        if token_id != str(image_token_id) and isinstance(item, dict) and item.get("content") == "<image>":
            del decoder[token_id]

    additional = [
        token for token in config.get("additional_special_tokens", []) if token not in {"<image>", f"<SPECIAL_{image_token_id}>"}
    ]
    additional.append("<image>")
    config["additional_special_tokens"] = additional
    config["image_token"] = "<image>"
    config["model_max_length"] = int(llava_next_config.get("tokenizer_model_max_length", config.get("model_max_length", 4096)))
    config["padding_side"] = llava_next_config.get("tokenizer_padding_side", config.get("padding_side", "right"))
    write_json(path, config)


def rewrite_special_tokens_map(path: Path, image_token_id: int) -> None:
    special = load_json(path)
    additional = special.setdefault("additional_special_tokens", [])
    normalized = []
    for item in additional:
        content = item.get("content") if isinstance(item, dict) else item
        if content not in {"<image>", f"<SPECIAL_{image_token_id}>"}:
            normalized.append(item)
    normalized.append(
        {
            "content": "<image>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        }
    )
    special["additional_special_tokens"] = normalized
    write_json(path, special)


def write_metadata(
    source_dir: Path,
    output_dir: Path,
    llava_next_config_path: Path,
    args: argparse.Namespace,
) -> LlavaAnythingConfig:
    llava_next_config = load_json(llava_next_config_path)
    config = build_llava_anything_config(
        llava_next_config=llava_next_config,
        source_dir=source_dir,
        text_model_name_or_path=args.text_model_name_or_path,
        vision_model_name_or_path=args.vision_model_name_or_path,
        vision_model_type=args.vision_model_type,
        vision_image_size=args.vision_image_size,
        vision_patch_size=args.vision_patch_size,
        vision_intermediate_size=args.vision_intermediate_size,
        image_token_id=args.image_token_id,
    )
    config.save_pretrained(output_dir)
    write_json(
        output_dir / "processor_config.json",
        build_processor_config(
            llava_next_config,
            config.vision_config.image_size,
            config.vision_config.patch_size,
            config.vision_feature_select_strategy,
            config.num_additional_image_tokens,
        ),
    )

    copy_if_exists(source_dir, output_dir, "tokenizer.json", prefer_backup=args.prefer_backups)
    copy_if_exists(source_dir, output_dir, "tokenizer_config.json", prefer_backup=args.prefer_backups)
    copy_if_exists(source_dir, output_dir, "special_tokens_map.json", prefer_backup=args.prefer_backups)
    copy_if_exists(source_dir, output_dir, "chat_template.jinja")
    copy_if_exists(source_dir, output_dir, "generation_config.json")

    rewrite_tokenizer_json(output_dir / "tokenizer.json", args.image_token_id)
    rewrite_tokenizer_config(output_dir / "tokenizer_config.json", args.image_token_id, llava_next_config)
    rewrite_special_tokens_map(output_dir / "special_tokens_map.json", args.image_token_id)
    return config


def dtype_from_arg(value: str) -> torch.dtype | str | None:
    if value == "auto":
        return "auto"
    if value.lower() in {"none", "null"}:
        return None
    dtype = getattr(torch, value, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}")


def convert_model_weights(source_dir: Path, output_dir: Path, config: LlavaAnythingConfig, args: argparse.Namespace) -> None:
    model_kwargs: dict[str, Any] = {
        "config": config,
        "key_mapping": dict(LlavaAnythingForConditionalGeneration._legacy_llava_next_key_mapping),
    }
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = dtype_from_arg(args.torch_dtype)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = LlavaAnythingForConditionalGeneration.from_pretrained(source_dir, **model_kwargs)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        save_original_format=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="LLaVA-NeXT checkpoint directory.")
    parser.add_argument("output_dir", type=Path, help="Directory for the converted LLaVA-Anything checkpoint.")
    parser.add_argument(
        "--llava-next-config",
        type=Path,
        help="Original LLaVA-NeXT config.json. Defaults to config.json.llava-next.bak when present.",
    )
    parser.add_argument("--text-model-name-or-path", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--vision-model-name-or-path")
    parser.add_argument("--vision-model-type", help="Defaults to clip_vision_model or siglip_vision_model inferred from mm_vision_tower.")
    parser.add_argument("--vision-image-size", type=int, help="Defaults to the size inferred from position embeddings.")
    parser.add_argument("--vision-patch-size", type=int, help="Defaults to the size inferred from patch embedding weights.")
    parser.add_argument("--vision-intermediate-size", type=int, help="Defaults to the size inferred from the vision MLP weights.")
    parser.add_argument("--image-token-id", type=int, default=DEFAULT_IMAGE_TOKEN_ID)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--metadata-only", action="store_true", help="Only write config, tokenizer, and processor files.")
    parser.add_argument("--prefer-backups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    llava_next_config_path = (
        args.llava_next_config.resolve()
        if args.llava_next_config is not None
        else existing_source_file(source_dir, "config.json", prefer_backup=args.prefer_backups).resolve()
    )

    prepare_output_dir(output_dir, args.overwrite)
    config = write_metadata(source_dir, output_dir, llava_next_config_path, args)
    if not args.metadata_only:
        convert_model_weights(source_dir, output_dir, config, args)
    print(f"Saved LLaVA-Anything conversion to {output_dir}")


if __name__ == "__main__":
    main()
