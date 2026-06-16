# Installation

This guide shows a generic local setup. The same package can be installed on a
laptop, workstation, cloud VM, or managed cluster as long as the Python and GPU
dependencies are available.

## Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch
- Transformers 5.8.1 or newer
- Enough CPU/GPU memory for the models you choose

For CUDA systems, install the PyTorch wheel that matches your CUDA version
before installing LLaVA-Anything.

## Basic Install

```bash
git clone https://github.com/naufalso/LLaVa-Anything.git
cd LLaVa-Anything

python -m venv llava-anything-env
source llava-anything-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check that the command-line tools are available:

```bash
llava-anything-build --help
llava-anything-train --help
llava-anything-infer --help
```

## Optional Extras

Install only the extras you need:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[wandb]"
python -m pip install -e ".[deepspeed]"
python -m pip install -e ".[eval]"
```

| Extra | Adds |
| --- | --- |
| `dev` | Test dependencies |
| `wandb` | Weights & Biases experiment logging |
| `deepspeed` | DeepSpeed integration through Hugging Face Trainer |
| `eval` | The `lmms-eval` model adapter entry point |

## Hugging Face Access

Some base models require accepting licenses or logging in to Hugging Face:

```bash
huggingface-cli login
```

Use normal Hugging Face cache environment variables such as `HF_HOME` or
`TRANSFORMERS_CACHE` if you want model files stored outside the default cache.

## Notes For Shared Infrastructure

Large training and evaluation jobs should be launched according to the rules of
your own compute environment. The package itself does not require a specific job
scheduler or cluster layout.
