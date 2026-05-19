# LLaVa-Anything Roadmap

## Purpose

LLaVa-Anything is a Hugging Face-native project for turning any compatible
Transformers causal LLM into a LLaVA-style vision-language model using a YAML
configuration. The project should eventually let a user define the base LLM,
vision encoder, multimodal projector, processor behavior, and training settings,
then get a unified model that works with Hugging Face Auto APIs and familiar
image-text inference patterns.

The current implementation is an MVP package skeleton. It proves the public
shape of the package, but it has not yet been validated on real GPU hardware
with full 8B-class models or trained projector weights.

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
- Unit tests and local CPU smoke tests.
- Architecture decision record:
  - `docs/architecture/adr-001-hf-native-composition.md`
- Handoff:
  - `docs/development/handoff-2026-05-19.md`

Verified locally:

- `uv run --extra dev pytest -q`
- YAML artifact build for Qwen3 + CLIP.
- Auto API smoke tests with tiny in-memory model configs.
- `generate()` with tiny image inputs.

Not verified yet:

- Real Qwen3-8B + CLIP component loading on NVIDIA.
- Real Apertus-8B-Instruct-2509 + SigLIP component loading on NVIDIA.
- Pipeline compatibility.
- Full model save/load with actual component weights.
- Training.

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

Goal: prove the package can instantiate real first-version target components on
NVIDIA hardware.

Targets:

- `Qwen/Qwen3-8B` + `openai/clip-vit-large-patch14-336`
- `swiss-ai/Apertus-8B-Instruct-2509` + `google/siglip-so400m-patch14-384`

Tasks:

- Capture environment info:
  - OS
  - GPU model and VRAM
  - NVIDIA driver
  - CUDA runtime
  - PyTorch version
  - Transformers version
  - Accelerate version
- Run unit tests.
- Build YAML artifacts without weights.
- Load full pretrained components with:
  - `torch_dtype=torch.bfloat16`
  - `device_map="auto"`
- Check actual class names for language and vision components.
- Run one forward pass with a real image and prompt.
- Run `generate()` with `max_new_tokens=8` first, then longer if stable.

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

- Both first-version target configs load their text and vision components.
- No dtype/device mismatch during forward.
- No image-token/image-feature mismatch for standard image sizes.
- Any failures are captured with stack traces and converted into tests.

Likely risks:

- `device_map="auto"` may not move the custom wrapper the same way it moves the
  nested models.
- `model.device` may be insufficient for multi-GPU dispatch.
- Qwen3 and Apertus may differ in generation kwargs expectations.
- SigLIP may return a different hidden-state shape than CLIP in some variants.

## Milestone 2 - Processor And Pipeline Compatibility

Goal: make the package feel like a normal Hugging Face image-text model.

Tasks:

- Validate `AutoProcessor.from_pretrained(...)`.
- Validate `AutoModelForImageTextToText.from_pretrained(...)`.
- Validate `pipeline("image-text-to-text", model=..., processor=...)`.
- Confirm `apply_chat_template` works for:
  - Qwen3 chat template
  - Apertus chat template
  - no chat template fallback
- Confirm image-token expansion counts match the model feature count.
- Add tests for saved processor reload from temporary directories.
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
- Pipeline either works or has a documented blocker with an issue/task.

## Milestone 3 - Full Save/Load Workflow

Goal: compose pretrained components, save the resulting VLM, and reload it
without custom build code.

Tasks:

- Extend builder with a documented command for full composition:

```bash
uv run llava-anything-build examples/qwen3_clip.yaml \
  --output-dir checkpoints/qwen3-clip-full \
  --load-pretrained-components
```

- Verify saved files include:
  - config
  - tokenizer
  - image processor
  - processor config
  - model weights
- Confirm `AutoModelForImageTextToText.from_pretrained` reloads the full model.
- Confirm additional image token additions resize language embeddings before
  saving.
- Add tests using tiny local models to avoid 8B dependencies.

Exit criteria:

- A composed model can be loaded with only `AutoProcessor` and
  `AutoModelForImageTextToText`.
- No direct call to `model_from_yaml_dict` is required for inference after
  saving.

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

- Add fixture that creates and saves a tiny text model.
- Add fixture that creates and saves a tiny vision model.
- Add fixture that creates tokenizer/image processor artifacts.
- Run builder against local file paths.
- Test full save/load/generate.

Exit criteria:

- CI can verify the core composition flow without GPU and without downloading
  giant models.

## Milestone 5 - Training Data Path

Goal: support LLaVA-style supervised fine-tuning for image-text instruction
data.

Tasks:

- Define training YAML schema separately from model YAML.
- Add dataset reader for LLaVA conversation JSON.
- Add image loading and preprocessing path.
- Add label masking for user tokens and image tokens.
- Add collator for variable-length image-expanded prompts.
- Add minimal `Trainer`/`SFTTrainer` entry point.
- Support trainable module selection:
  - projector only
  - projector + vision tower
  - projector + LoRA on language model
  - full fine-tune
- Add PEFT dependency only as optional extra if used.

Exit criteria:

- A tiny synthetic image-text dataset can overfit on a tiny model.
- Projector-only training runs end to end.
- Loss decreases on the synthetic task.

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
| Qwen/Qwen3-8B | openai/clip-vit-large-patch14-336 | Pending GPU | Pending GPU | Pending GPU | Pending GPU | Pending GPU | Pending GPU | First target |
| swiss-ai/Apertus-8B-Instruct-2509 | google/siglip-so400m-patch14-384 | Pending GPU | Pending GPU | Pending GPU | Pending GPU | Pending GPU | Pending GPU | First target |

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
- docs/development/handoff-2026-05-19.md
- docs/development/roadmap.md
- docs/architecture/adr-001-hf-native-composition.md

Current phase: NVIDIA GPU validation and first real-model integration.

Use uv by default. Do not modify or include LLaVA-NeXT; it is reference-only.
Keep the public names llava-anything, llava_anything, llava_anything model_type,
and LlavaAnything* classes.

Start by running:
uv venv .venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run llava-anything-build examples/qwen3_clip.yaml --output-dir checkpoints/qwen3-clip-config
uv run llava-anything-build examples/apertus_siglip.yaml --output-dir checkpoints/apertus-siglip-config

Then validate full pretrained component loading on NVIDIA with bfloat16 and
device_map="auto" for both example YAML files. Capture environment info,
commands, failures, and fixes. Convert failures into tests where practical.
```

