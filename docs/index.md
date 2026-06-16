# Documentation

Welcome to the LLaVA-Anything docs. These guides are organized around the path a
new user usually takes: understand the project, install it, build a first model,
train on custom data, run inference, and evaluate checkpoints.

## Start Here

| Page | Use it when you want to |
| --- | --- |
| [Background](background.md) | Understand the design goals and the main components |
| [Installation](installation.md) | Set up the package and optional extras |
| [Quick Start](quickstart.md) | Build a first artifact and run a small inference check |
| [Configuration](configuration.md) | Write model and training YAML files |
| [Training](training.md) | Train a projector or finetune the full multimodal model |
| [Inference](inference.md) | Use the CLI or Transformers APIs for generation |
| [Evaluation](evaluation.md) | Connect saved checkpoints to evaluation frameworks |
| [Roadmap](roadmap.md) | See what is stable now and what is planned next |

## Design Notes

- [ADR-001: Use Hugging Face-Native Composition](architecture/adr-001-hf-native-composition.md)

## Documentation Principles

The public docs avoid assuming a specific machine, dataset location, scheduler,
or research project. Examples use local paths such as `data/` and
`checkpoints/`, but you can replace those with any paths that make sense for
your workstation, cloud instance, or cluster.
