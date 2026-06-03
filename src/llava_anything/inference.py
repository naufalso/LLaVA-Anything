"""Quick inference utilities for saved LLaVa-Anything checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything  # noqa: F401 - registers Auto classes

from .training import _conversation_text, _load_json_records

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user "
    "provides, and assist the user with a variety of tasks using natural language."
)


def _torch_dtype(value: str) -> torch.dtype | str | None:
    if value == "auto":
        return "auto"
    if value in {"none", "None"}:
        return None
    dtype = getattr(torch, value, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise argparse.ArgumentTypeError(f"Unknown torch dtype: {value}")


def _load_image(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _render_prompt(processor: Any, prompt: str, system_prompt: str | None = None) -> str:
    image_token = getattr(processor, "image_token", "<image>")
    content: str | list[dict[str, str]]
    if image_token in prompt:
        content = prompt
    else:
        content = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": content})
    return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def _record_prompt(record: dict[str, Any]) -> str:
    user_text, _ = _conversation_text(record)
    return user_text


def _record_image_path(record: dict[str, Any], image_folder: str | Path) -> Path:
    image_name = record.get("image")
    if not image_name:
        raise ValueError("Evaluation records must include an image path.")
    return Path(image_folder) / str(image_name)


def _move_inputs_to_device(inputs: Any, device: torch.device | str) -> Any:
    return inputs.to(device)


def _decode_generated_text(processor: Any, output: torch.Tensor, input_length: int) -> str:
    generated_ids = output[input_length:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def generate_response(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 128,
) -> str:
    rendered_prompt = _render_prompt(processor, prompt, system_prompt)
    inputs = processor(images=image, text=rendered_prompt, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return _decode_generated_text(processor, output[0], inputs["input_ids"].shape[-1])


def run_single_image(args: argparse.Namespace, model: Any, processor: Any) -> None:
    result = generate_response(
        model=model,
        processor=processor,
        image=_load_image(args.image_input),
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


def run_dataset(args: argparse.Namespace, model: Any, processor: Any) -> None:
    records = _load_json_records(args.data_path)
    if args.sample >= 0:
        records = records[: args.sample]

    for index, record in enumerate(records):
        image_path = _record_image_path(record, args.image_folder)
        prompt = _record_prompt(record)
        try:
            prediction = generate_response(
                model=model,
                processor=processor,
                image=_load_image(image_path),
                prompt=prompt,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
            )
        except FileNotFoundError:
            prediction = ""
            error = f"Image not found: {image_path}"
        else:
            error = None

        item = {
            "index": index,
            "id": record.get("id"),
            "image": str(record.get("image", "")),
            "prompt": prompt,
            "target": _conversation_text(record)[1],
            "prediction": prediction,
        }
        if error is not None:
            item["error"] = error
        print(json.dumps(item, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Directory containing a saved LLaVa-Anything model.")
    parser.add_argument("--image-input", type=Path, default=Path("examples/image/example-image1.jpg"))
    parser.add_argument("--prompt", default="Describe this image")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--data-path", type=Path, help="JSON or JSONL data path for multi-image evaluation.")
    parser.add_argument("--image-folder", type=Path, help="Root folder for image paths in --data-path records.")
    parser.add_argument("--sample", type=int, default=10, help="Number of records to evaluate. Use -1 for all records.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--torch-dtype", default="auto", type=_torch_dtype)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_kwargs = {}
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    if args.data_path is not None and args.image_folder is not None:
        run_dataset(args, model, processor)
    else:
        run_single_image(args, model, processor)


if __name__ == "__main__":
    main()
