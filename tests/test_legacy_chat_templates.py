from __future__ import annotations

from pathlib import Path

import llava_anything  # noqa: F401 - registers AutoProcessor classes
import pytest
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAVA_V1_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
)


@pytest.mark.parametrize("model_dir", ["models/llava-v1.5-7b", "models/PALO-7B"])
def test_converted_vicuna_llava_models_use_llava_v1_chat_template(model_dir: str) -> None:
    model_path = REPO_ROOT / model_dir
    if not model_path.exists():
        pytest.skip(f"Local converted checkpoint is not present: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_path)
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}],
        }
    ]

    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    assert rendered == f"{LLAVA_V1_SYSTEM_PROMPT}USER: <image>\nDescribe the image. ASSISTANT:"
