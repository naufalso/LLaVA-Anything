# LLaVa-Anything

**Build LLaVA-style vision-language models from Hugging Face language and vision models.**

LLaVa-Anything is a Hugging Face-native project for composing vision-language
models from standard Transformers components. Define a base causal LLM, a vision
encoder, and a multimodal projector in YAML, then build a model that can be used
with familiar Hugging Face `AutoProcessor` and `AutoModelForImageTextToText`
APIs.

The goal is to make VLM experimentation easier when you want to try a new
Hugging Face LLM that is not already supported by an existing LLaVA
implementation. Instead of starting from a large fork and tracing model-specific
changes across the codebase, LLaVa-Anything keeps the composition layer small,
explicit, and reusable.

## Why This Project?

LLaVA has made multimodal LLM research much more accessible, but adapting the
framework to a new language model can still take a lot of custom glue code. This
project explores a cleaner path: keep the text model, vision tower, projector,
and processor as independent pieces, then connect them through a shared
Hugging Face-style interface.

The first version is intentionally practical. It focuses on image-text models,
YAML-based configuration, Hugging Face save/load behavior, and projector
training before expanding into broader model support.

## What It Supports Today

- YAML configs for choosing the language model, vision tower, projector, and
  processor behavior.
- A shared LLaVA-style model wrapper built from Transformers components.
- Hugging Face `AutoConfig`, `AutoProcessor`, and
  `AutoModelForImageTextToText` loading.
- CLIP and SigLIP-style vision towers.
- Stage-1 projector pretraining with frozen LLM and vision tower.
- Stage-2 full model finetuning.
- Validation scripts for component loading and image-text smoke tests.

Initial target LLMs:

- `swiss-ai/Apertus-8B-Instruct-2509`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-1.7B` for lower-memory validation

Initial target vision towers:

- CLIP-style Transformers vision models
- SigLIP-style Transformers vision models

## How It Works

A model is assembled from four parts:

- a Hugging Face causal language model
- a Hugging Face vision encoder
- a multimodal projector
- a processor that prepares image-text prompts

The YAML builder creates the config and processor artifacts for this composed
model. You can save only the config/processor for later loading, or save a full
local checkpoint with pretrained language and vision weights.

## Install

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

## Build From YAML

```bash
uv run llava-anything-build examples/qwen3_clip.yaml --output-dir checkpoints/qwen3-clip-vlm
```

This creates a config and processor. Pass `--load-pretrained-components` when
you want to materialize the base LLM and vision weights locally before saving:

```bash
uv run llava-anything-build examples/qwen3_clip.yaml \
  --output-dir checkpoints/qwen3-clip-full \
  --load-pretrained-components
```

A full saved artifact can be reloaded with `AutoProcessor` and
`AutoModelForImageTextToText` without calling the YAML builder at inference time.
Transformers 5.8 stores the image processor payload inside
`processor_config.json` for this composite processor.

## Inference

For quick checkpoint testing, use the bundled inference command:

```bash
uv run llava-anything-infer checkpoints/qwen3-clip-vlm
```

By default this reads `examples/image/example-image1.jpg` and prompts with
`Describe this image`. Override either value as needed:

```bash
uv run llava-anything-infer checkpoints/qwen3-clip-vlm \
  --image-input examples/image/example-image2.png \
  --prompt "What text is visible in this image?"
```

The same command can evaluate LLaVA-style JSON or JSONL records with the
`image` and `conversations` fields used by training. Dataset mode only runs
when both `--data-path` and `--image-folder` are provided. It evaluates 10
records by default and prints one JSON object per record:

```bash
uv run llava-anything-infer checkpoints/qwen3-clip-vlm \
  --data-path data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --image-folder data/LLaVA-Pretrain
```

Use `--sample -1` to evaluate every record:

```bash
uv run llava-anything-infer checkpoints/qwen3-clip-vlm \
  --data-path data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --image-folder data/LLaVA-Pretrain \
  --sample -1
```

For custom scripts, saved artifacts can also be loaded directly with
Transformers:

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

## Stage-1 Projector Pretraining

Stage-1 follows the LLaVA feature-alignment setup: the language model and
vision tower are frozen, and only `multi_modal_projector` is trained. The
training configuration is separate from the model YAML.

First download and extract the LLaVA-Pretrain dataset:

```bash
scripts/download_llava_pretrain.sh
```

Then start projector pretraining:

```bash
uv run llava-anything-train examples/qwen3_1_7b_clip_base_pretrain.yaml
```

For LLaVA-NeXT-style any-resolution packing with the same Qwen3 1.7B +
CLIP ViT-B/32 components, use the anyres config:

```bash
uv run llava-anything-train examples/qwen3-1.7b/qwen3_1_7b_clip_base_anyres_pretrain.yaml
```

`llava-anything-pretrain` is still available as a backward-compatible alias for
older Stage-1 commands.

The provided example expects LLaVA-Pretrain annotations at
`data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json` and images under
`data/LLaVA-Pretrain`. It uses `examples/qwen3_1_7b_clip_base.yaml`, bfloat16,
and projector-only training. Set `training.bf16: false` and remove
`model_kwargs.torch_dtype` if you are doing a tiny CPU-only smoke.

Set `logging.preview_samples` to print rendered prompt/target examples before
training starts. Weights & Biases is optional and only activates when the
training YAML defines a top-level `wandb:` block:

```bash
uv pip install -e ".[dev,wandb]"
```

```yaml
wandb:
  project: llava-anything
  name: qwen3-1.7b-clip-base-pretrain-projector
  mode: offline
```

## Stage-2 Full Model Finetuning

Stage-2 starts from the Stage-1 projector checkpoint and finetunes the full
composed model on LLaVA-Instruct data. The dataset is not included in this repo,
so prepare the instruction JSON and images before running Stage-2.

Stage-2 full finetuning has been completed and tested with
`examples/qwen3_1_7b_clip_base_stage2_full.yaml` on `qwen3-1.7b-clip-base`.

The Stage-2 configs expect this layout:

```text
data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
data/LLaVA-Instruct-150K/coco/train2017/
data/LLaVA-Instruct-150K/gqa/images/
data/LLaVA-Instruct-150K/ocr_vqa/images/
data/LLaVA-Instruct-150K/textvqa/train_images/
data/LLaVA-Instruct-150K/vg/VG_100K/
data/LLaVA-Instruct-150K/vg/VG_100K_2/
```

Download or place `llava_v1_5_mix665k.json` under `data/LLaVA-Instruct-150K`,
then download and extract the image datasets:

```bash
scripts/download_llava_instruct_images.sh
```

For a short validation run before the full finetune, use the smoke config. It
reads the same instruction JSON, filters records to images that exist locally,
and limits the run to a small sample:

```bash
uv run llava-anything-train examples/qwen3_1_7b_clip_base_stage2_smoke.yaml
```

After the full dataset is ready, run the full Stage-2 config:

```bash
uv run llava-anything-train examples/qwen3_1_7b_clip_base_stage2_full.yaml
```

Both configs expect the Stage-1 checkpoint at
`checkpoints/qwen3-1.7b-clip-base-pretrain-projector`.

For the any-resolution Stage-2 path, first run the anyres Stage-1 command
above, then use:

```bash
uv run llava-anything-train examples/qwen3_1_7b_clip_base_anyres_stage2_smoke.yaml
uv run llava-anything-train examples/qwen3_1_7b_clip_base_anyres_stage2_full.yaml
```

The anyres Stage-2 configs expect the Stage-1 checkpoint at
`checkpoints/qwen3-1.7b-clip-base-anyres-pretrain-projector`.

## Validation Scripts

Load real YAML components on a GPU machine:

```bash
uv run python scripts/validate_gpu_components.py examples/qwen3_clip.yaml
```

Run a saved-model image-text smoke through Hugging Face Auto APIs:

```bash
uv run python scripts/smoke_image_text_generation.py checkpoints/qwen3-clip-full --image image.jpg
```

The projector starts randomly initialized unless you have trained or loaded
projector weights, so these smokes validate runtime compatibility rather than
answer quality.

## Current Status

LLaVa-Anything is an alpha project under active development. The main
composition path is in place, and the first training and validation utilities are
available. Image understanding quality depends on trained projector or adaptor
weights, so early smoke tests are mainly used to confirm that the runtime path is
working.

For the detailed engineering roadmap, see
[`docs/development/roadmap.md`](docs/development/roadmap.md).

## Roadmap

- [x] Hugging Face-native model and processor composition.
- [x] YAML-based VLM configuration.
- [x] Hugging Face Auto API save/load support.
- [x] CLIP and SigLIP-style vision tower support.
- [x] Stage-1 projector pretraining.
- [x] Stage-2 full model finetuning, tested on `qwen3-1.7b-clip-base`.
- [x] Basic validation and smoke-test scripts.
- [ ] Broader validation across more LLMs and vision towers.
- [ ] More ready-to-run example configs.
- [ ] LLaVA-NeXT-style any-resolution image handling.
- [ ] More documentation for custom model integration.
- [ ] Community compatibility reports.
- [ ] Low priority: publish a unified Hugging Face dataset mirror for training data.

## Contributing

Contributions are welcome, especially around new model compatibility, example
configs, training recipes, validation results, and documentation. If you try a
new LLM or vision tower, please share what worked, what failed, and the main
environment details such as Transformers, PyTorch, CUDA, and GPU version.
