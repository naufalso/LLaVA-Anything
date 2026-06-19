# Configuration

LLaVA-Anything uses YAML for both model construction and training. Model YAML
describes the components. Training YAML describes what to load, which data to
use, and which parameters should be trainable.

## Model YAML

A minimal model YAML has three main sections:

```yaml
model:
  model_type: llava_anything
  image_token: "<image>"
  vision_feature_layer: -2
  vision_feature_select_strategy: default
  image_seq_length: 576
  projector_type: mlp2x_gelu
  projector_hidden_act: gelu

text_model:
  name_or_path: your-org/your-causal-lm
  trust_remote_code: false
  tokenizer:
    padding_side: left
    model_max_length: 2048

vision_model:
  name_or_path: openai/clip-vit-large-patch14-336
  trust_remote_code: false
  image_processor:
    patch_size: 14
    num_additional_image_tokens: 1
```

### `model`

| Field | Purpose |
| --- | --- |
| `image_token` | Placeholder token inserted where image features should appear |
| `vision_feature_layer` | Vision tower layer used for image features |
| `vision_feature_select_strategy` | Feature selection strategy passed to the model |
| `image_seq_length` | Fixed image-token count for `fixed` mode |
| `projector_type` | Projector architecture, such as `linear` or `mlp2x_gelu` |
| `projector_hidden_act` | Activation used by MLP projectors |

### `text_model`

Use any compatible Transformers causal LM. Set `trust_remote_code: true` only
when the base model requires it and you trust the model repository.

### `vision_model`

Use a compatible Transformers vision model and image processor. The image
processor settings should match the selected vision tower.

For OpenCLIP vision towers, set `backend: open_clip`. LLaVA-Anything reads the
repository's `open_clip_config.json`, builds an OpenCLIP visual tower through
`open_clip.create_model`, and mirrors the OpenCLIP preprocessing metadata with a
Transformers `CLIPImageProcessor`.

```yaml
vision_model:
  backend: open_clip
  name_or_path: chs20/fare2-clip
  image_processor:
    patch_size: 14
    num_additional_image_tokens: 1
```

Install `llava-anything[openclip]` before loading pretrained OpenCLIP
components. Hub model ids such as `chs20/fare2-clip` are normalized to
OpenCLIP's `hf-hub:` format automatically; local OpenCLIP directories are
normalized to `local-dir:`.

## Any-Resolution Images

Add an `image` section when you want LLaVA-NeXT-style any-resolution packing:

```yaml
image:
  mode: anyres
  anyres:
    enabled: true
    grid_pinpoints:
      - [224, 224]
      - [224, 448]
      - [448, 224]
      - [448, 448]
```

Use `mode: fixed` or omit the `image` section for the simpler fixed-token path.

## Training YAML

Training YAML points to either a model YAML or an existing checkpoint.

Stage 1 usually starts from `model_yaml` and trains only the projector:

```yaml
model_yaml: path/to/model.yaml
load_pretrained_components: true
model_kwargs:
  torch_dtype: bfloat16

data:
  data_path: data/train.jsonl
  image_folder: data/images

training:
  output_dir: checkpoints/my-vlm-stage1
  trainable_modules: projector
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  learning_rate: 1.0e-3
  gradient_checkpointing: true
  resume_from_checkpoint: true
  model_max_length: 2048
  bf16: true
  save_strategy: steps
  save_steps: 500
  remove_unused_columns: false
```

For Hugging Face Datasets, replace the local JSON fields with a Hub source:

```yaml
data:
  hf_dataset_path: naufalso/LLaVA-Pretrain
  hf_dataset_name: pretrain
  hf_dataset_split: train
```

`image_folder` is only needed for HF datasets whose `image` values are relative
paths rather than decoded or embedded image records.

Stage 2 usually starts from a checkpoint and trains more modules:

```yaml
model_checkpoint: checkpoints/my-vlm-stage1
model_kwargs:
  torch_dtype: bfloat16

data:
  data_path: data/instruction.jsonl
  image_folder: data/images
  available_images_only: true

training:
  output_dir: checkpoints/my-vlm-stage2
  trainable_modules: full
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  num_train_epochs: 1
  learning_rate: 2.0e-5
  gradient_checkpointing: true
  resume_from_checkpoint: true
  model_max_length: 2048
  bf16: true
  remove_unused_columns: false
```

The `training` section is passed to Hugging Face `TrainingArguments` after
LLaVA-Anything removes its own keys such as `trainable_modules`.

## Trainable Modules

`training.trainable_modules` accepts a comma-separated list:

| Value | Trains |
| --- | --- |
| `projector` | The multimodal projector only |
| `vision_tower` | The vision encoder |
| `language_model` | The language model |
| `full` | Projector, vision tower, and language model |

You can combine module names, for example:

```yaml
training:
  trainable_modules: projector,language_model
```

## Optional Logging

Weights & Biases activates only when a top-level `wandb` section is present:

```yaml
wandb:
  project: llava-anything
  name: my-vlm-stage1
  mode: online
```
