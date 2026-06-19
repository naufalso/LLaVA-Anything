# Evaluation

LLaVA-Anything aims to make evaluation adapters thin and reusable. The saved
checkpoint should remain a normal Hugging Face-style artifact, while benchmark
frameworks provide task definitions, prompting rules, judging, and reporting.

## Support Levels

| Framework | Status | Use it for |
| --- | --- | --- |
| `lmms-eval` | Built-in adapter through the `eval` extra | Multimodal generation benchmarks supported by lmms-eval |
| `VLMEvalKit` | Compatible through a small external adapter | VLMEvalKit datasets, judges, and reports |
| `lm-evaluation-harness` | Use for text-only evaluation of the base language model, or add a local adapter for custom multimodal tasks | Language-model benchmarks and custom task suites |
| `HarmBench` | Local fork checkout under `eval/HarmBench` | Text-only and multimodal safety evaluation |

The repo does not vendor external evaluation projects. Install them in the way
their upstream maintainers recommend, then point them at saved LLaVA-Anything
checkpoints.

Upstream projects:

- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HarmBench](https://github.com/naufalso/HarmBench)

## lmms-eval

Install the evaluation extra:

```bash
python -m pip install -e ".[eval]"
```

Check that the adapter is registered:

```bash
python - <<'PY'
from lmms_eval.models import get_model_manifest

manifest = get_model_manifest("llava_anything")
print(manifest.model_id)
print(manifest.simple_class_path)
PY
```

Run a task supported by your installed `lmms-eval` version:

```bash
accelerate launch -m lmms_eval \
  --model llava_anything \
  --model_args pretrained=checkpoints/my-vlm,dtype=bfloat16,batch_size=1 \
  --tasks <task_name> \
  --batch_size 1
```

Common model arguments:

| Argument | Meaning |
| --- | --- |
| `pretrained` | Path or Hub ID of a saved LLaVA-Anything checkpoint |
| `dtype` | Torch dtype, such as `auto`, `bfloat16`, or `float16` |
| `batch_size` | Adapter batch size per process |
| `device` | Device string, usually `cuda` or `cpu` |
| `device_map` | Transformers device map, usually `auto` |
| `max_new_tokens` | Default generation limit |
| `system_prompt` | Optional system prompt string |
| `trust_remote_code` | Whether to trust remote model code |

Task names, dataset downloads, judging modes, and metrics come from
`lmms-eval`; choose the tasks that match your model and research question.

## VLMEvalKit

VLMEvalKit uses its own model registry. To evaluate a LLaVA-Anything checkpoint,
install VLMEvalKit following its upstream guide, then add a small adapter in
your VLMEvalKit checkout that loads:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git external/VLMEvalKit
python -m pip install -r external/VLMEvalKit/requirements.txt
python -m pip install -e external/VLMEvalKit
```

The adapter should use the same loading pattern as normal LLaVA-Anything
inference:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

import llava_anything

processor = AutoProcessor.from_pretrained(checkpoint_path)
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

Then adapt VLMEvalKit's expected `generate` method to call the processor,
prepare the image and prompt, and return generated text. Keep this adapter small
and independent so it can be upstreamed or maintained alongside your evaluation
project.

## lm-evaluation-harness

The EleutherAI `lm-evaluation-harness` is strongest for text-only language-model
evaluation. For a multimodal project, it is useful in two ways:

- evaluate the base language model before multimodal training
- evaluate text-only capabilities of a finetuned checkpoint when you provide an
  adapter that exposes the harness model interface

Install the harness from upstream:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness external/lm-evaluation-harness
python -m pip install -e external/lm-evaluation-harness
python -m pip install "lm_eval[hf]"
```

For image-aware tasks, use a multimodal evaluation framework such as
`lmms-eval` or VLMEvalKit unless you are intentionally building a custom harness
task.

## HarmBench

This workspace expects the editable HarmBench fork at `eval/HarmBench`:

```bash
git clone https://github.com/naufalso/HarmBench.git eval/HarmBench
python -m pip install -r eval/HarmBench/requirements.txt
```

For test-case generation, completions, and classifier scoring in this
workspace, use the dedicated `.venv-harmbench` environment. It pins the
Torch/CUDA line used by the project and avoids HarmBench's broad
`vllm>=0.3.0` resolver path by using the local Hugging Face classifier backend:

```bash
UV_CACHE_DIR=.cache/uv uv venv .venv-harmbench --python 3.12
UV_CACHE_DIR=.cache/uv uv pip install --python .venv-harmbench/bin/python -e . \
  "torch==2.6.0" "torchvision==0.21.0" "transformers==5.8.1" \
  pandas tqdm numpy accelerate pillow pyyaml datasketch "spacy==3.7.2" \
  sentencepiece tiktoken protobuf
UV_CACHE_DIR=.cache/uv uv pip install --python .venv-harmbench/bin/python \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

The local HarmBench fork still supports the upstream vLLM classifier path, but
the project launcher defaults `CLASSIFIER_BACKEND=hf` so `.venv-harmbench` can
run step 3 without installing vLLM or Ray.

The local fork includes two LLaVA-Anything model entries in
`eval/HarmBench/configs/model_configs/models.yaml`:

| HarmBench model | Use it for | Required environment variable |
| --- | --- | --- |
| `llava_anything` | Multimodal HarmBench completions with a saved LLaVA-Anything checkpoint | `LLAVA_ANYTHING_MODEL_PATH` |
| `llava_anything_text` | Text-only HarmBench completions with the underlying language model or text-only checkpoint | `LLAVA_ANYTHING_TEXT_MODEL_PATH` |
| `llava_anything_vlm_text` | Text-only HarmBench completions loaded from a full LLaVA-Anything VLM checkpoint | `LLAVA_ANYTHING_TEXT_MODEL_PATH` |

For SLURM runs, use the project launcher from the repository root:

```bash
sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
```

Useful overrides:

```bash
TARGET=multimodal sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
TARGET=text sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
TARGET=both STEP=2_and_3 sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
TARGET=text STEP=3 CLASSIFIER_BACKEND=hf sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
```

Text-only runs can target either the language model directly:

```bash
LLAVA_ANYTHING_TEXT_MODEL_PATH=swiss-ai/Apertus-8B-Instruct-2509 \
TARGET=text STEP=2 TEXT_MODEL_NAME=llava_anything_text \
  sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
```

or the full VLM checkpoint in text-only mode:

```bash
LLAVA_ANYTHING_TEXT_MODEL_PATH="${PWD}/models/llava-1.5-apertus-8b-clipl-en" \
TARGET=text STEP=2 TEXT_MODEL_NAME=llava_anything_vlm_text \
  sbatch scripts/sbatch/eval/eval_llava_anything_harmbench_1gpu.sbatch
```

The launcher activates `.venv-harmbench`, sources `.env` when present, adds both
`src/` and `eval/HarmBench` to `PYTHONPATH`, and runs inside one H200 GPU
allocation. By default it uses `MultiModalDirectRequest` for multimodal
evaluation, `DirectRequest` for text-only evaluation, and the Hugging Face
classifier backend for step 3. Override `METHODS`, `TEXT_METHODS`, `MODEL_NAME`,
`TEXT_MODEL_NAME`, `MAX_NEW_TOKENS`, `CLS_PATH`, `CLASSIFIER_BACKEND`,
`CLASSIFIER_BATCH_SIZE`, `CLASSIFIER_DTYPE`, and `CLASSIFIER_DEVICE_MAP` when
you need different HarmBench pipeline settings.

The LLaVA-Anything HarmBench wrapper supports completion generation. It does
not implement the gradient loss required by HarmBench's multimodal PGD attacks.

## Reporting

Keep evaluation outputs separate from training checkpoints. A simple layout is:

```text
outputs/
  evaluations/
    my-vlm/
      lmms-eval/
      vlmevalkit/
      text-only/
```

Record the checkpoint path, git revision, task names, framework version, model
arguments, and decoding settings with every run. That metadata is more useful
for comparison than a fixed list of benchmark names in the project docs.
