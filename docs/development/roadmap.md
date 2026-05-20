# LLaVa-Anything Roadmap

## Purpose

LLaVa-Anything is a Hugging Face-native project for turning any compatible
Transformers causal LLM into a LLaVA-style vision-language model using a YAML
configuration. The project should eventually let a user define the base LLM,
vision encoder, multimodal projector, processor behavior, and training settings,
then get a unified model that works with Hugging Face Auto APIs and familiar
image-text inference patterns.

The current implementation is an MVP HF-native package that has passed local
unit tests plus NVIDIA GPU runtime smokes for Apertus/SigLIP and a lower-memory
Qwen3/CLIP-base combination. The projector is still randomly initialized, so
image-text generation is a runtime/API check rather than a quality signal.

## Product Goals

- Provide a single compositional VLM wrapper instead of one Python subclass per
  language model family.
- Support Hugging Face `AutoConfig`, `AutoProcessor`, and
  `AutoModelForImageTextToText` loading.
- Let users create VLM configs from YAML.
- Support Qwen3 and Apertus in the first working version.
- Support CLIP and SigLIP-style vision towers in the first working version.
- Preserve a LLaVA-like architecture: vision tower, projector, image-token
  expansion, and causal LLM generation.
- Keep `LLaVA-NeXT/` as reference-only material, not part of the GitHub project.

## Non-Goals For The First Version

- Full LLaVA-NeXT OneVision parity.
- Video or multi-image training.
- High-quality zero-shot image understanding before projector training.
- Serving infrastructure such as vLLM, TGI, FastAPI, or Gradio.
- A broad benchmark suite.
- Custom kernels or performance optimization beyond standard Transformers,
  Accelerate, and PyTorch behavior.

## Current State

Implemented:

- `llava_anything` package.
- `LlavaAnythingConfig`.
- `LlavaAnythingForConditionalGeneration`.
- `LlavaAnythingProcessor`.
- YAML builder CLI: `llava-anything-build`.
- Auto registration for config, processor, and image-text model loading.
- Example YAML files:
  - `examples/qwen3_clip.yaml`
  - `examples/apertus_siglip.yaml`
  - `examples/qwen3_1_7b_clip_base.yaml` for lower-memory GPU smoke tests
  - `examples/qwen3_1_7b_clip_base_stage2_smoke.yaml` for partial-data Stage-2 smoke tests
  - `examples/qwen3_1_7b_clip_base_stage2_full.yaml` for full Stage-2 finetuning
- Tokenizer YAML options including `padding_side` and `model_max_length`.
- Unit and smoke regression tests for:
  - config serialization
  - image-token expansion
  - image-token/image-feature mismatch errors
  - `generate()` with tiny image inputs
  - tokenizer vocab handling for pretrained checkpoints with reserved vocab rows
  - wrapper device/dtype behavior under dispatched pretrained components
  - runtime metadata dependencies needed for GPU validation
  - saved processor reload through `AutoProcessor`
  - saved full tiny composed model reload through `AutoModelForImageTextToText`
  - `pipeline("image-text-to-text")` dict and chat inputs with a tiny local model and processor
- Committed validation scripts:
  - `scripts/validate_gpu_components.py`
  - `scripts/smoke_image_text_generation.py`
- Committed data preparation scripts:
  - `scripts/download_llava_pretrain.sh`
- Architecture decision record:
  - `docs/architecture/adr-001-hf-native-composition.md`

Verified on NVIDIA GPU, May 19, 2026:

- Environment:
  - GPU: NVIDIA GeForce RTX 4090, 24 GiB VRAM
  - NVIDIA driver: `535.288.01`
  - PyTorch: `2.6.0+cu124`
  - CUDA runtime reported by PyTorch: `12.4`
  - Transformers: `5.8.1`
  - Accelerate: `1.13.0`
- `uv run pytest -q`: `13 passed`.
- Config-only artifact builds for:
  - `examples/qwen3_clip.yaml`
  - `examples/apertus_siglip.yaml`
  - `examples/qwen3_1_7b_clip_base.yaml`
- `AutoConfig.from_pretrained(...)`.
- `AutoProcessor.from_pretrained(...)`.
- `AutoModelForImageTextToText.from_config(...)`.
- Full pretrained component loading with `torch_dtype=torch.bfloat16` and
  `device_map="auto"` for:
  - `swiss-ai/Apertus-8B-Instruct-2509` + `google/siglip-so400m-patch14-384`
  - `Qwen/Qwen3-1.7B` + `openai/clip-vit-base-patch32`
- Minimal local image-text `generate(max_new_tokens=8)` smoke tests for:
  - Apertus + SigLIP
  - Qwen3-1.7B + CLIP-base

Issues found during GPU validation and fixed:

- `device_map="auto"` requires `accelerate`; added it as a runtime dependency.
- Plain `torch>=2.2` allowed incompatible/newer CUDA wheels and too-old
  `torch.load` behavior for Transformers 5.8; constrained runtime metadata to
  Python `>=3.10,<3.13`, `torch>=2.6,<2.7`, and `torchvision>=0.21,<0.22`.
- Processor construction was shrinking or growing pretrained text vocab metadata
  before checkpoint loading; it now preserves checkpoint vocab size and lets model
  loading resize embeddings only when required by the added image token.
- Wrapper `device`/`dtype` followed CPU-only wrapper parameters instead of the
  dispatched language model; it now follows the language input embeddings.
- Image feature extraction now sends image tensors through the vision tower
  device/dtype and returns projected features on the language embedding device.
- Added-image-token embedding resize now runs before loading the vision tower and
  disables memory-heavy mean resizing, avoiding unnecessary CUDA OOM on 24 GiB
  cards.

Known validation notes:

- `Qwen/Qwen3-8B` + `openai/clip-vit-large-patch14-336` was started but not
  completed because unauthenticated HF Hub shard downloads were too slow for the
  session. The lower-memory Qwen3-1.7B + CLIP-base path validates the same
  runtime API path.
- `HF_HUB_DISABLE_XET=1` was useful on this machine to avoid stalled Xet-backed
  downloads.
- Apertus emits an optional xIELU warning when the custom CUDA kernel is not
  installed; it falls back successfully.
- Projector weights are still randomly initialized, so generated text is only an
  API/runtime smoke and not a quality signal.

Still not verified:

- `pipeline("image-text-to-text")` compatibility for full 8B target checkpoints on GPU.
- Full composed model save/load with actual 8B target component weights.
- Full Qwen3-8B + CLIP-large load/generate after model shards are locally cached.
- Training or fine-tuning.

## Architecture Direction

The accepted architecture is HF-native composition:

- `LlavaAnythingConfig` owns nested `text_config` and `vision_config`.
- `LlavaAnythingForConditionalGeneration` owns:
  - `language_model: AutoModelForCausalLM`
  - `vision_tower: AutoModel`
  - `multi_modal_projector`
- `LlavaAnythingProcessor` owns:
  - tokenizer
  - image processor
  - image-token expansion behavior
- YAML builder creates config and processor artifacts, and can optionally load
  pretrained components.

Architecture should remain simple until real tests prove a need for special
cases. Avoid reintroducing per-LLM wrapper classes unless a concrete model
cannot be supported through standard Transformers interfaces.

## Milestone 0 - Repository Baseline

Goal: make the project portable and unambiguous on any machine.

Tasks:

- Confirm `LLaVA-NeXT/` is ignored and not staged.
- Confirm old `llava_everything` names are gone.
- Confirm `uv.lock` resolves on Linux/NVIDIA.
- Add a short `CONTRIBUTING.md` if multiple machines or contributors will work
  on the project.
- Add basic CI later when the repo is pushed.

Validation:

```bash
uv venv .venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run python -m compileall src tests
```

Exit criteria:

- Fresh clone can install and run tests without referencing `LLaVA-NeXT/`.
- `git status --short` shows only intentional project files.

## Milestone 1 - NVIDIA GPU Runtime Validation

Status: mostly complete. Apertus/SigLIP and the lower-memory Qwen3/CLIP-base
smoke path pass on a 24 GiB RTX 4090. Full Qwen3-8B + CLIP-large still needs a
rerun after weights are cached or on a faster/authenticated HF Hub connection.

Goal: prove the package can instantiate real first-version target components on
NVIDIA hardware.

Targets:

- `Qwen/Qwen3-8B` + `openai/clip-vit-large-patch14-336`
- `Qwen/Qwen3-1.7B` + `openai/clip-vit-base-patch32`
- `swiss-ai/Apertus-8B-Instruct-2509` + `google/siglip-so400m-patch14-384`

Completed tasks:

- Captured GPU, driver, CUDA runtime, PyTorch, Transformers, and Accelerate
  versions.
- Ran unit tests.
- Built YAML artifacts without weights.
- Loaded full pretrained components with:
  - `torch_dtype=torch.bfloat16`
  - `device_map="auto"`
- Checked actual class names for language and vision components.
- Ran short local-image `generate()` smoke tests for Apertus/SigLIP and
  Qwen3-1.7B/CLIP-base.
- Converted GPU validation failures into regression tests where practical.

Remaining tasks:

- Complete full Qwen3-8B + CLIP-large loading and generation after weights are
  cached.
- Add a committed validation script instead of relying on one-off `/tmp` scripts.
- Run a longer generation smoke after projector/adaptor weights exist.

Validation script outline:

```python
import torch
from llava_anything.builder import load_yaml, config_from_yaml_dict, model_from_yaml_dict

data = load_yaml("examples/qwen3_clip.yaml")
config = config_from_yaml_dict(data)
model = model_from_yaml_dict(
    data,
    config=config,
    load_pretrained_components=True,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
)
print(model.language_model.__class__.__name__)
print(model.vision_tower.__class__.__name__)
print(model.hf_device_map if hasattr(model, "hf_device_map") else "no device map")
```

Exit criteria:

- Apertus/SigLIP loads text and vision components on NVIDIA. Done.
- Qwen3-1.7B/CLIP-base loads text and vision components on NVIDIA. Done.
- Qwen3-8B/CLIP-large loads text and vision components on NVIDIA. Pending
  cached weights or faster/authenticated download.
- No dtype/device mismatch during forward or short generation. Done for tested
  combos.
- No image-token/image-feature mismatch for standard image sizes. Done for
  tested combos.
- Any failures are captured with stack traces and converted into tests where
  practical. Done for dependency, vocab, resize, and wrapper device/dtype bugs.

Resolved risks:

- `device_map="auto"` does not move CPU-only wrapper parameters, so wrapper
  `device`/`dtype` now follow the language input embeddings.
- Processor vocab mutation before checkpoint loading broke Qwen3 and Apertus;
  checkpoint vocab is now preserved until after pretrained weights load.
- Adding a new image token to Apertus required embedding resize; resize now runs
  before loading the vision tower and disables memory-heavy mean resizing.

Remaining risks:

- Multi-GPU dispatch still needs explicit validation.
- Qwen3-8B + CLIP-large still needs full validation after weights are cached.
- Pipeline compatibility is still unknown.

## Milestone 2 - Processor And Pipeline Compatibility

Goal: make the package feel like a normal Hugging Face image-text model.

Tasks:

- Validate `AutoProcessor.from_pretrained(...)`.
- Validate `AutoModelForImageTextToText.from_pretrained(...)`.
- Validate `pipeline("image-text-to-text", model=..., processor=...)`. Done for tiny local dict and chat-style inputs.
- Confirm `apply_chat_template` works for:
  - Qwen3 chat template
  - Apertus chat template
  - no chat template fallback
- Confirm image-token expansion counts match the model feature count.
- Add tests for saved processor reload from temporary directories. Done.
- Document the expected prompt format.

Validation examples:

```python
import llava_anything
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("checkpoints/qwen3-clip-config")
model = AutoModelForImageTextToText.from_pretrained(
    "checkpoints/qwen3-clip-full",
    torch_dtype="auto",
    device_map="auto",
)
```

Exit criteria:

- Auto APIs work after saving config, processor, and model weights.
- Pipeline either works or has a documented blocker with an issue/task. Done for tiny local dict and chat-style coverage; full target GPU smokes remain.

## Milestone 3 - Full Save/Load Workflow

Goal: compose pretrained components, save the resulting VLM, and reload it
without custom build code.

Tasks:

- Extend builder with a documented command for full composition. Done:

```bash
uv run llava-anything-build examples/qwen3_clip.yaml \
  --output-dir checkpoints/qwen3-clip-full \
  --load-pretrained-components
```

- Verify saved files include. Done for tiny local full artifacts:
  - config
  - tokenizer
  - image processor payload in `processor_config.json`
  - processor config
  - model weights
- Confirm `AutoModelForImageTextToText.from_pretrained` reloads the full model. Done for tiny local full artifacts.
- Confirm additional image token additions resize language embeddings before
  saving. Done for tiny local full artifacts.
- Add tests using tiny local models to avoid 8B dependencies. Done.

Exit criteria:

- A composed model can be loaded with only `AutoProcessor` and
  `AutoModelForImageTextToText`. Done for tiny local components; full target GPU validation remains.
- No direct call to `model_from_yaml_dict` is required for inference after
  saving. Done for tiny local full artifacts.

## Milestone 4 - Tiny Integration Models

Goal: create fast integration tests that exercise real Transformers loading
without requiring 8B models.

Candidate text models:

- tiny Llama-style causal model from a local generated config.
- small public causal LM if network use is acceptable.

Candidate vision models:

- tiny CLIP config generated locally.
- small public CLIP-like vision tower if network use is acceptable.

Tasks:

- Add fixture that creates and saves a tiny text model. Done.
- Add fixture that creates and saves a tiny vision model. Done.
- Add fixture that creates tokenizer/image processor artifacts. Done.
- Run builder against local file paths. Done.
- Test full save/load/generate. Done.

Exit criteria:

- CI can verify the core composition flow without GPU and without downloading
  giant models. Done with local tiny generated components.

## Milestone 5 - Training Data Path

Goal: support LLaVA-style supervised fine-tuning for image-text instruction
data.

Tasks:

- Define training YAML schema separately from model YAML. Done for stage-1 projector pretraining.
- Add dataset reader for LLaVA conversation JSON. Done for JSON/JSONL LLaVA-style records.
- Add image loading and preprocessing path. Done for single-image records.
- Add label masking for user tokens and image tokens. Done for assistant-only loss.
- Add collator for variable-length image-expanded prompts. Done.
- Add minimal `Trainer`/`SFTTrainer` entry point. Done with `Trainer` via `llava-anything-train` (`llava-anything-pretrain` remains as a compatibility alias).
- Add checkpoint-based training resume for Stage-2. Done with `model_checkpoint` training YAML.
- Add available-image filtering for partial LLaVA-Instruct downloads. Done with `data.available_images_only`.
- Add LLaVA-Pretrain download/extract helper. Done with `scripts/download_llava_pretrain.sh`.
- Support trainable module selection:
  - projector only. Done.
  - projector + vision tower. Basic selection supported; full validation pending.
  - projector + LoRA on language model. Pending optional PEFT work.
  - full fine-tune. Stage-2 configs added; full validation pending.
- Add PEFT dependency only as optional extra if used. Pending.

Exit criteria:

- A tiny synthetic image-text dataset can overfit on a tiny model. Initial decreasing-loss smoke done.
- Projector-only training runs end to end. Done on tiny generated components.
- Loss decreases on the synthetic task. Done.
- LLaVA-Pretrain data can be prepared with one script, producing
  `data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json` and extracted image
  shard directories such as `data/LLaVA-Pretrain/00000/`.
- Stage-2 can start from a Stage-1 composed checkpoint and use the
  LLaVA-Instruct mix JSON with either available-image filtering for smoke runs
  or the full downloaded image set for full finetuning.

## Milestone 6 - Projector And Adapter Improvements

Goal: make the multimodal adapter configurable enough for real experiments.

Current projector support:

- `linear`
- `mlpNx_gelu`
- `identity`

Potential additions:

- gated MLP projector
- residual MLP projector
- layer norm before/after projection
- Q-Former-style resampler
- Perceiver-style resampler
- pretrained projector loading

Tasks:

- Decide which projector types are needed for v1.
- Document projector schema in YAML.
- Add strict validation for incompatible hidden sizes.
- Add load/save tests for projector-only checkpoints.

Exit criteria:

- Projector configs are documented and validated.
- Pretrained projector weights can be loaded independently from the LLM.

## Milestone 7 - LLaVA-NeXT Any-Resolution Support

Goal: replace fixed image-token counts with LLaVA-NeXT-style image packing when
enabled.

Tasks:

- Implement image grid pinpoints in config.
- Compute per-image feature counts from original image sizes.
- Preserve base image features plus unpadded high-resolution features.
- Add newline feature insertion.
- Update processor expansion to match variable image sizes.
- Add tests for:
  - square image
  - wide image
  - tall image
  - multiple image sizes in a batch
- Keep fixed-token mode available for simpler CLIP/SigLIP experiments.

Exit criteria:

- Processor image-token expansion and model image-feature packing agree for all
  supported image sizes.
- Any-resolution mode works with CLIP and SigLIP.

## Milestone 8 - Trust Remote Code Policy

Goal: safely support models requiring custom Transformers code.

Tasks:

- Keep `trust_remote_code` explicit in YAML.
- Print/log a warning when enabled.
- Add tests with a small remote-code model if feasible.
- Document security implications.
- Ensure config save/load preserves trust flags.

Exit criteria:

- Remote-code support is deliberate, documented, and tested on at least one real
  model or clearly marked experimental.

## Milestone 9 - Inference UX

Goal: make common inference flows easy and documented.

Tasks:

- Add `examples/inference_from_yaml.py`.
- Add `examples/inference_from_saved_model.py`.
- Add `examples/build_full_model.py`.
- Add README sections for:
  - config-only build
  - full model build
  - AutoProcessor + AutoModel inference
  - pipeline inference
  - expected random-output behavior before training
- Add image fixture or documented external image path.

Exit criteria:

- A new user can run a documented inference smoke test without reading source.

## Milestone 10 - Evaluation And Quality Checks

Goal: establish lightweight correctness checks before adding heavy benchmarks.

Tasks:

- Add shape/dtype/device checks after model loading.
- Add image-token count diagnostics.
- Add `--dry-run` builder mode.
- Add a script that prints config summary:
  - text model type
  - vision model type
  - image sequence length
  - projector type
  - image token id
- Add a tiny golden test for deterministic generation with tiny random models.

Exit criteria:

- Most integration failures can be diagnosed from one command output.

## Milestone 11 - Packaging And Release

Goal: make the project usable as a standalone GitHub package.

Tasks:

- Add license file if not already present.
- Add `CONTRIBUTING.md`.
- Add GitHub Actions CPU test workflow.
- Add issue templates for:
  - model compatibility
  - vision tower compatibility
  - training bug
  - feature request
- Add changelog.
- Add versioning policy.

Exit criteria:

- Fresh clone, install, tests, and examples are documented and reproducible.

## YAML Schema Roadmap

Current examples use this rough shape:

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
  name_or_path: Qwen/Qwen3-8B
  trust_remote_code: false
  tokenizer:
    padding_side: left
    model_max_length: 1024

vision_model:
  name_or_path: openai/clip-vit-large-patch14-336
  trust_remote_code: false
  image_processor:
    patch_size: 14
    num_additional_image_tokens: 1
```

Planned additions:

```yaml
adapter:
  pretrained_path: null
  freeze: false

training:
  trainable_parts:
    - projector
  lora:
    enabled: false
  data:
    format: llava_json
    path: data/train.json
    image_root: data/images

image:
  mode: fixed
  anyres:
    enabled: false
    grid_pinpoints: null
```

Schema principles:

- Keep model construction YAML separate from experiment/training YAML if the
  training schema grows large.
- Validate unknown fields rather than silently ignoring them.
- Keep `trust_remote_code` explicit and local to each component.

## Compatibility Matrix

Track each tested model combination in a table like this:

| Text model | Vision model | Config build | Component load | Forward | Generate | Pipeline | Save/load | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-8B | openai/clip-vit-large-patch14-336 | Pass | Download-bound | Not run | Not run | Pending | Pending | First target; shard fetch was too slow unauthenticated during validation session |
| Qwen/Qwen3-1.7B | openai/clip-vit-base-patch32 | Pass | Pass | Pass via generate prefill | Pass | Pending | Pending | Lower-memory Qwen runtime smoke; used `HF_HUB_DISABLE_XET=1` |
| swiss-ai/Apertus-8B-Instruct-2509 | google/siglip-so400m-patch14-384 | Pass | Pass | Pass via generate prefill | Pass | Pending | Pending | First target; optional xIELU kernel warning is non-fatal |

Update this matrix after every serious compatibility test.

## Testing Strategy

Use four layers of tests:

1. Unit tests:
   - config serialization
   - processor expansion
   - projector construction
   - placeholder mismatch errors

2. Tiny integration tests:
   - tiny local text model
   - tiny local vision model
   - save/load/generate

3. GPU smoke tests:
   - full first-target model loading
   - dtype/device behavior
   - short generation

4. Training smoke tests:
   - tiny synthetic dataset
   - projector-only overfit
   - LoRA path if implemented

Every bug found in a higher layer should become a lower-layer regression test
where practical.

## Documentation Roadmap

Required docs before broader sharing:

- README quickstart.
- YAML schema reference.
- GPU validation guide.
- Model compatibility matrix.
- Training guide.
- Adapter/projector guide.
- Troubleshooting guide.

Troubleshooting should include:

- image token count mismatch
- CUDA out of memory
- `device_map` dispatch issues
- missing `accelerate` for `device_map="auto"`
- incompatible PyTorch/CUDA wheel for the installed NVIDIA driver
- missing `torchvision`
- tokenizer has no pad token
- chat template issues
- random outputs before training

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `device_map="auto"` fails with nested wrapper | Blocks large model loading | Add explicit dispatch tests; use Accelerate helpers if needed |
| Image-token expansion diverges from vision output length | Runtime failure | Add diagnostics and tests for every image mode |
| Pipeline expects metadata not present on custom model | Pipeline incompatibility | Compare with HF LLaVA-NeXT model metadata and add missing attributes |
| Remote-code models have nonstandard forward signatures | Model-specific breakage | Keep compatibility shims small and tested |
| Random projector makes inference look broken | User confusion | Document that training/projector weights are required |
| Any-resolution packing gets complex | Bugs in image layout | Keep fixed mode and add anyres incrementally |

## Decision Log

Accepted:

- Use HF-native composition.
- Keep `LLaVA-NeXT/` reference-only.
- Use `llava_anything` and `llava-anything` names.
- First target LLMs are Qwen3-8B and Apertus-8B-Instruct-2509.
- First target vision towers are CLIP and SigLIP.
- First implementation uses fixed image-token counts.

Pending:

- Whether any-resolution support is required before first release.
- Whether training should use plain `Trainer`, TRL, or a custom loop.
- Whether PEFT should be a core dependency or optional extra.
- How to package pretrained projector checkpoints.

## Next Agent Prompt

Use this prompt when continuing elsewhere:

```text
You are continuing LLaVa-Anything development.

Read these first:
- docs/development/roadmap.md
- docs/architecture/adr-001-hf-native-composition.md

Current phase: processor/pipeline compatibility and full saved-model workflow.
NVIDIA GPU validation on May 19, 2026 passed for Apertus/SigLIP and
Qwen3-1.7B/CLIP-base. Full Qwen3-8B/CLIP-large still needs a rerun after model
weights are cached or with an authenticated/faster HF Hub connection.

Use uv by default. Do not modify or include LLaVA-NeXT; it is reference-only.
Keep the public names llava-anything, llava_anything, llava_anything model_type,
and LlavaAnything* classes. Do not reintroduce llava_everything names.

Start by running:
uv venv .venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run llava-anything-build examples/qwen3_clip.yaml --output-dir checkpoints/qwen3-clip-config
uv run llava-anything-build examples/apertus_siglip.yaml --output-dir checkpoints/apertus-siglip-config
uv run llava-anything-build examples/qwen3_1_7b_clip_base.yaml --output-dir checkpoints/qwen3-1.7b-clip-base-config

Recommended next work:
1. Run `pipeline("image-text-to-text")` and saved-model smokes against the full GPU target artifacts.
2. Rerun Qwen3-8B/CLIP-large once weights are available locally.
3. Add training/fine-tuning path design once the inference contract is stable.
```
