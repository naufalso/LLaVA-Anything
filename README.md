# LLaVA-Anything

Build LLaVA-style vision-language models from the Hugging Face models you
already know.

LLaVA-Anything is a small, composable framework for turning a causal language
model plus a vision encoder into a multimodal model. You describe the pieces in
YAML, train the projector or the full model when you are ready, and save
artifacts that load through familiar Transformers APIs.

The project is for researchers, builders, and open-source teams who want a
clear starting point for their own VLM or MLLM without maintaining a large
model-family-specific fork.

## Why This Exists

Many multimodal projects are tightly coupled to a small set of language models,
vision towers, datasets, and launch environments. That is useful for reproducing
a paper, but it can be difficult when you want to try a different LLM, swap the
vision tower, train on your own data, or package the result for other users.

LLaVA-Anything keeps the core idea explicit:

- choose a Hugging Face causal LLM
- choose a Hugging Face vision encoder
- connect them with a multimodal projector
- train with LLaVA-style image and conversation data
- load the result with Hugging Face `Auto*` classes

## What You Can Do

- Build model configs and processors from YAML.
- Assemble full checkpoints from pretrained text and vision components.
- Use fixed-token image handling or LLaVA-NeXT-style any-resolution image
  packing.
- Train a projector-only Stage 1 model, then optionally finetune the full
  multimodal model.
- Run single-image or JSON/JSONL batch inference.
- Evaluate saved checkpoints through the built-in `lmms-eval` adapter, with
  guidance for other evaluation frameworks.

## Installation

LLaVA-Anything supports Python 3.10, 3.11, and 3.12.

```bash
git clone https://github.com/naufalso/LLaVa-Anything.git
cd LLaVa-Anything

python -m venv llava-anything-env
source llava-anything-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional extras only when you need them:

```bash
python -m pip install -e ".[dev]"        # tests and development
python -m pip install -e ".[wandb]"      # Weights & Biases logging
python -m pip install -e ".[deepspeed]"  # DeepSpeed training support
python -m pip install -e ".[eval]"       # lmms-eval integration
```

For GPU training or inference, install the PyTorch build that matches your CUDA
or accelerator environment before installing the project.

See [Installation](docs/installation.md) for more detail.

## Quick Start

Build a starter model artifact from YAML:

```bash
llava-anything-build \
  examples/qwen3-1.7b/qwen3_1_7b_clip_base.yaml \
  --output-dir checkpoints/example-vlm
```

That command creates a Hugging Face-style config and processor. To also load and
save the pretrained text and vision weights into the output directory, add
`--load-pretrained-components`:

```bash
llava-anything-build \
  examples/qwen3-1.7b/qwen3_1_7b_clip_base.yaml \
  --output-dir checkpoints/example-vlm-full \
  --load-pretrained-components
```

Run inference after you have a checkpoint with trained multimodal weights:

```bash
llava-anything-infer checkpoints/example-vlm-full \
  --image-input examples/image/example-image1.jpg \
  --prompt "Describe this image."
```

If the projector has not been trained yet, the model can load successfully but
its image understanding will not be meaningful. Start with the training guide
when you are building a new VLM from base components.

## Common Workflows

Train from a YAML recipe:

```bash
llava-anything-train path/to/training.yaml
```

Load a saved checkpoint in Python:

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything  # registers LLaVA-Anything Auto classes

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
            {"type": "text", "text": "What is happening in this image?"},
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

Evaluate with `lmms-eval` after installing the `eval` extra:

```bash
accelerate launch -m lmms_eval \
  --model llava_anything \
  --model_args pretrained=checkpoints/my-vlm,dtype=bfloat16,batch_size=1 \
  --tasks <task_name> \
  --batch_size 1
```

## Documentation

- [Documentation Home](docs/index.md)
- [Background and Concepts](docs/background.md)
- [Installation](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [Configuration](docs/configuration.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Evaluation](docs/evaluation.md)
- [Roadmap](docs/roadmap.md)
- [Architecture Decision Record](docs/architecture/adr-001-hf-native-composition.md)

## Project Status

LLaVA-Anything is early-stage software. The core composition, training, inference,
and first evaluation adapter are in place, but APIs may still evolve as the
project adds broader model support, packaging, examples, and evaluation
integrations.

Contributions that improve usability, documentation, model compatibility, test
coverage, or clean evaluation adapters are especially welcome.
