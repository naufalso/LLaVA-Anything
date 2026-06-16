# Quick Start

This guide builds a first LLaVA-Anything artifact and shows the basic inference
flow. It is meant as a smoke test, not a training recipe.

## 1. Install The Package

```bash
python -m pip install -e .
```

## 2. Build From YAML

Use one of the example model configs:

```bash
llava-anything-build \
  examples/qwen3-1.7b/qwen3_1_7b_clip_base.yaml \
  --output-dir checkpoints/example-vlm
```

This writes the model config and processor to `checkpoints/example-vlm`.

To also save the pretrained language and vision weights into the same artifact,
use:

```bash
llava-anything-build \
  examples/qwen3-1.7b/qwen3_1_7b_clip_base.yaml \
  --output-dir checkpoints/example-vlm-full \
  --load-pretrained-components
```

The full build downloads the base text and vision models. Pick smaller
components in your YAML if you are working on limited hardware.

## 3. Run Inference

After you have a trained checkpoint, run:

```bash
llava-anything-infer checkpoints/example-vlm-full \
  --image-input examples/image/example-image1.jpg \
  --prompt "Describe this image."
```

If you only assembled pretrained components and have not trained the multimodal
projector, the command can still run but the response should not be treated as a
meaningful VLM result.

## 4. Move To Training

To train your own model, prepare image-text records and a training YAML, then
run:

```bash
llava-anything-train path/to/training.yaml
```

Continue with [Training](training.md) for the data format and recommended
stages.
