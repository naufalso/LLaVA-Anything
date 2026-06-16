# Training

Training is driven by YAML and standard Hugging Face `Trainer` arguments. The
recommended path is two stages:

1. Train the multimodal projector while the language model and vision tower are
   frozen.
2. Finetune the full model, or a selected subset of modules, on instruction
   data.

## Data Format

The trainer reads JSON or JSONL files containing LLaVA-style records. Each
record should include an `image` path relative to `data.image_folder` and a
conversation with at least one user turn and one assistant turn.

```json
{
  "id": "example-0001",
  "image": "images/example.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe the image."
    },
    {
      "from": "gpt",
      "value": "A person is standing beside a bicycle."
    }
  ]
}
```

The role names `human`/`gpt` and `user`/`assistant` are both supported.

Hugging Face Datasets with the same record fields are also supported. Use
`data.hf_dataset_path` instead of `data.data_path`; `data.image_folder` is not
required when the dataset has a decoded or embedded `Image` column. For the
LLaVA pretraining dataset on the Hub:

```yaml
data:
  hf_dataset_path: naufalso/LLaVA-Pretrain
  hf_dataset_name: pretrain
  hf_dataset_split: train
```

`hf_dataset_name` is the Hugging Face dataset config name and can be omitted for
single-config datasets. `hf_dataset_config` and `hf_dataset_config_name` are
accepted aliases. Use `hf_dataset_revision` to pin a branch, tag, or commit.
If a Hub dataset stores image values as relative strings instead of an `Image`
column, continue to provide `image_folder` so those paths can be resolved.

## Stage 1: Projector Training

Stage 1 aligns visual features with the language model. It is usually cheaper
than full finetuning because only the projector is trainable.

```yaml
model_yaml: path/to/model.yaml
load_pretrained_components: true
model_kwargs:
  torch_dtype: bfloat16

data:
  data_path: data/pretrain.jsonl
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
  logging_steps: 10
  save_strategy: steps
  save_steps: 500
  remove_unused_columns: false
```

Run it with:

```bash
llava-anything-train path/to/stage1.yaml
```

The recommended defaults use gradient checkpointing to reduce activation memory,
resume automatically from the latest checkpoint in `training.output_dir`, and
cap training sequences at 2048 tokens. Raise `model_max_length` only when your
model, data, and hardware can support the longer context.

## Stage 2: Instruction Finetuning

Stage 2 starts from the Stage 1 checkpoint. Use `trainable_modules: full` for
full finetuning, or choose a subset when you want a lighter run.

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
  logging_steps: 10
  save_strategy: steps
  save_steps: 500
  remove_unused_columns: false
```

Run it with:

```bash
llava-anything-train path/to/stage2.yaml
```

## Useful Data Options

| Field | Purpose |
| --- | --- |
| `max_samples` | Limit the number of records for a smoke run |
| `available_images_only` | Skip records whose image files are missing |
| `available_images_cache_dir` | Store skip lists for faster reruns |
| `require_image` | Drop text-only records |
| `min_image_width` / `min_image_height` | Filter small images |
| `max_image_aspect_ratio` | Filter extreme aspect ratios |
| `max_image_tokens` | Filter records that would exceed an image-token budget |
| `system_prompt` | Add a system prompt during supervised training |

## Distributed And Large-Model Training

The trainer uses Hugging Face `TrainingArguments`, so standard launchers such as
`torchrun`, `accelerate launch`, and DeepSpeed-compatible Trainer configs can be
used according to your environment.

Example with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m llava_anything.training path/to/training.yaml
```

Example training YAML with DeepSpeed enabled:

```yaml
training:
  output_dir: checkpoints/my-vlm
  trainable_modules: full
  deepspeed: configs/deepspeed/zero2_bf16.json
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  resume_from_checkpoint: true
  model_max_length: 2048
  bf16: true
  remove_unused_columns: false
```

Choose launcher settings that match your own hardware and scheduling system.
