# Roadmap

This roadmap describes the public direction of LLaVA-Anything. It is not tied
to one branch, dataset, checkpoint, or private run.

## Stable Foundation

- Hugging Face-native model composition.
- YAML model builder.
- Combined tokenizer and image processor.
- Fixed-token and any-resolution image handling.
- Projector-only and full-model training through Hugging Face Trainer.
- Single-image and JSON/JSONL inference CLI.
- Saved checkpoints that load through Transformers `Auto*` APIs.
- Built-in `lmms-eval` adapter entry point.

## Near-Term Priorities

- Add more beginner-friendly example configs with smaller public models.
- Improve validation errors for incompatible model and processor combinations.
- Add Hub publishing guidance and checkpoint cards.
- Expand tests around any-resolution packing, generation, and training resumes.
- Provide first-class, documented adapters for more evaluation frameworks.
- Add LoRA and other parameter-efficient finetuning recipes.
- Improve examples for custom datasets and multilingual data.

## Longer-Term Ideas

- Multi-image conversations.
- Video or frame-sequence inputs.
- Quantized inference and training recipes.
- Model cards and dataset cards generated from training metadata.
- A small model zoo of community-contributed starter checkpoints.
- More projector architectures and vision feature selection strategies.

## Contribution Areas

Helpful contributions include:

- new model compatibility fixes
- clearer docs and examples
- small reproducible training recipes
- evaluation adapters that can be maintained without local patches
- tests for edge cases in image packing, chat templates, and checkpoint loading
