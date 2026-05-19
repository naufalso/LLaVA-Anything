#!/usr/bin/env python
"""Run a saved LLaVa-Anything image-text generation smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything  # noqa: F401 - registers Auto classes


def _torch_dtype(value: str) -> torch.dtype | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}") from exc


def _load_image(path: str | None) -> Image.Image:
    if path is None:
        return Image.new("RGB", (336, 336), color="white")
    return Image.open(Path(path)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory containing a fully saved LLaVa-Anything model")
    parser.add_argument("--image", help="Optional image path. Uses a blank RGB image when omitted.")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--torch-dtype", default="auto", type=_torch_dtype)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model_kwargs = {}
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **model_kwargs)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=_load_image(args.image), text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
