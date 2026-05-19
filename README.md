# LLaVa-Anything

LLaVa-Anything is a fresh Hugging Face-style package for building LLaVA-like
vision-language models from standard Transformers components.

The v1 goal is narrow and practical:

- use a YAML file to choose a base causal LLM, a vision encoder, and a projector
- compose those components into one `PreTrainedModel`
- support Hugging Face-style processor/model inference
- avoid one wrapper subclass per LLM family

The initial target LLMs are:

- `swiss-ai/Apertus-8B-Instruct-2509`
- `Qwen/Qwen3-8B`

The initial target vision towers are CLIP and SigLIP-style Transformers vision
models.

## Install

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

## Build From YAML

```bash
llava-anything-build examples/qwen3_clip.yaml --output-dir checkpoints/qwen3-clip-vlm
```

This creates a config and processor. Pass `--load-pretrained-components` when
you want to materialize the base LLM and vision weights locally before saving.

## Inference Shape

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything

model_id = "checkpoints/qwen3-clip-vlm"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
image = Image.open("image.jpg")
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128)

print(processor.decode(output[0], skip_special_tokens=True))
```

## Status

This is a first-version package skeleton. It implements the Hugging Face-native
composition point and fixed-token image encoding for CLIP/SigLIP-style image
towers. Training scripts and LLaVA-NeXT any-resolution packing should be added
after the model/processor contract is validated.

