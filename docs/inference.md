# Inference

LLaVA-Anything checkpoints can be used through the provided CLI or directly
with Transformers.

## CLI

Run a single image:

```bash
llava-anything-infer checkpoints/my-vlm \
  --image-input path/to/image.jpg \
  --prompt "Describe this image."
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--system-prompt` | Override the default system prompt |
| `--max-new-tokens` | Limit generated tokens |
| `--torch-dtype` | Set dtype, such as `auto`, `bfloat16`, or `float16` |
| `--device-map` | Pass a Transformers device map, such as `auto` |

## JSON Or JSONL Records

The CLI can also run over LLaVA-style records:

```bash
llava-anything-infer checkpoints/my-vlm \
  --data-path data/eval.jsonl \
  --image-folder data/images \
  --sample 100
```

Use `--sample -1` to process every record. The command prints one JSON object
per record with the prompt, generated text, and any loading error.

## Python

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything

model_id = "checkpoints/my-vlm"
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
image = Image.open("path/to/image.jpg").convert("RGB")
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128)

print(processor.decode(generated[0], skip_special_tokens=True))
```

## Practical Notes

- Use a checkpoint whose projector has been trained for image understanding.
- Use left padding for batched generation when your tokenizer supports it.
- Set `trust_remote_code=True` only for model repositories you trust.
- FlashAttention is used automatically by the CLI when the package is available.
