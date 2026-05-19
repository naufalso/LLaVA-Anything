# ADR-001: Use Hugging Face-Native Composition

## Status

Accepted

## Context

LLaVA-NeXT is used as reference only and will not be included in this project.
The new project needs to turn Hugging Face Transformers-based causal LLMs into
LLaVA-style VLMs through YAML configuration. It should support the Hugging Face
processor/model inference shape used by LLaVA-NeXT, including
`image-text-to-text` pipelines and AutoModel loading.

The first version targets a small-team research/development workflow, latest
Transformers, image-text support, and the following initial components:

- LLMs: `swiss-ai/Apertus-8B-Instruct-2509`, `Qwen/Qwen3-8B`
- vision towers: CLIP and SigLIP-style Transformers vision models
- adapters: linear and MLP projectors

## Options Considered

| Option | Pros | Cons | Complexity | When Valid |
| --- | --- | --- | --- | --- |
| Keep LLaVA-NeXT class-per-LLM wrappers | Fast reuse of existing code | Does not scale to arbitrary LLMs; keeps name-based branching | Medium | Maintaining a fork of LLaVA-NeXT |
| Wrap Hugging Face LLaVA-NeXT as-is | Lowest inference risk | Harder to customize adapters and generic LLM loading | Low | Only reproducing LLaVA-NeXT checkpoints |
| HF-native composition | Aligns with AutoConfig/AutoModel APIs; avoids per-LLM subclasses | Requires careful generation and processor integration | Medium | Building a general package |

## Decision

Use a Hugging Face-native compositional architecture:

- `LlavaAnythingConfig` stores nested `text_config` and `vision_config`.
- `LlavaAnythingForConditionalGeneration` composes an
  `AutoModelForCausalLM` language model, an `AutoModel` vision tower, and a
  multimodal projector.
- `LlavaAnythingProcessor` wraps a tokenizer and image processor and expands
  image placeholders to the expected number of image feature tokens.
- YAML is compiled into Hugging Face config/processor artifacts.
- `trust_remote_code` is supported through explicit config/YAML flags where
  Transformers can resolve the remote model.

## Rationale

This matches the goal of turning any compatible Transformers causal LLM into a
VLM without adding a new Python subclass for every language model family. It
also keeps model loading, saving, processor behavior, and pipeline integration
close to Hugging Face conventions.

## Trade-offs

- The v1 model starts with fixed image-token counts rather than full LLaVA-NeXT
  any-resolution packing.
- Remote-code LLMs depend on the base model exposing a compatible
  `AutoModelForCausalLM` interface.
- Training scripts are intentionally left as a later layer so the public
  model/processor contract can stabilize first.

## Consequences

- Positive: Qwen3, Apertus, and future supported LLMs can share one VLM wrapper.
- Positive: generated checkpoints can be consumed through Hugging Face Auto APIs.
- Negative: some custom LLMs may need model-specific compatibility fixes.
- Mitigation: keep `trust_remote_code` explicit and add small compatibility
  adapters only when a real target model proves it is needed.

## Revisit Trigger

Reconsider this decision if Transformers exposes a stable generic multimodal
composition API that makes this wrapper unnecessary, or if training requirements
force deep model-family-specific behavior into the core model.

