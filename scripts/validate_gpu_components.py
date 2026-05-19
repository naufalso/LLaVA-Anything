#!/usr/bin/env python
"""Validate full LLaVa-Anything component loading on a CUDA machine."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import __version__ as transformers_version

from llava_anything.builder import config_from_yaml_dict, load_yaml, model_from_yaml_dict


def _torch_dtype(value: str) -> torch.dtype | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}") from exc


def _format_device_map(model: Any) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        return "no device map"
    return str(device_map)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_path", help="YAML config to compose, for example examples/qwen3_clip.yaml")
    parser.add_argument("--torch-dtype", default="bfloat16", type=_torch_dtype)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    data = load_yaml(args.yaml_path)
    config = config_from_yaml_dict(data)
    model_kwargs: dict[str, Any] = {}
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = model_from_yaml_dict(
        data,
        config=config,
        load_pretrained_components=True,
        model_kwargs=model_kwargs,
    )

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers_version}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"cuda_runtime: {torch.version.cuda}")
    print(f"language_model: {model.language_model.__class__.__name__}")
    print(f"vision_tower: {model.vision_tower.__class__.__name__}")
    print(f"model_dtype: {model.dtype}")
    print(f"model_device: {model.device}")
    print(f"device_map: {_format_device_map(model)}")


if __name__ == "__main__":
    main()
