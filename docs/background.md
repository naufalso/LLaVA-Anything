# Background

LLaVA-Anything exists to make VLM construction feel modular instead of
model-specific. The project follows the LLaVA recipe, but keeps the language
model, vision tower, projector, processor, and training recipe as separate
pieces that can be swapped through configuration.

## The Problem

Adapting a LLaVA-style model to a new language model often means copying a
large codebase and changing model-family-specific branches until the new model
loads. That makes experimentation slow and makes it harder for other people to
reuse the result.

LLaVA-Anything takes a different approach: if a language model can load as a
Transformers causal LM and a vision encoder can load as a Transformers vision
model, the framework should be able to compose them with a projector and a
processor contract.

## Core Components

- `text_model`: the base causal language model and tokenizer.
- `vision_model`: the image encoder and image processor.
- `multi_modal_projector`: the module that maps visual features into the
  language model hidden space.
- `processor`: the combined tokenizer and image processor used for prompts and
  image tensors.
- `image` settings: fixed-token or any-resolution image packing behavior.
- `training` YAML: data paths, trainable modules, and Hugging Face
  `TrainingArguments`.

## Workflow

1. Choose a text model and vision tower.
2. Write a model YAML that describes the components.
3. Build a Hugging Face-style artifact with `llava-anything-build`.
4. Train the multimodal projector on image-text data.
5. Optionally finetune the full model.
6. Save the result and load it with Transformers `AutoProcessor` and
   `AutoModelForImageTextToText`.

## Image Modes

`fixed` mode uses one fixed image-token budget. It is simple and useful for
early experiments.

`anyres` mode follows the LLaVA-NeXT-style idea of packing images into a grid of
possible resolutions. It is useful when your data contains images with varied
aspect ratios or when you want to preserve more visual detail.

## What This Project Is Not

LLaVA-Anything is not tied to one benchmark suite, one dataset mirror, one
cluster, or one saved checkpoint. The repository provides examples and adapters,
but the main goal is to give you a clean foundation for your own VLM or MLLM.
