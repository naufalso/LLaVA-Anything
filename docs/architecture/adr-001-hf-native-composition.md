# ADR-001: Use Hugging Face-Native Composition

## Status

Accepted

## Context

LLaVA-style systems combine a language model, a vision encoder, a projector, and
a processor contract. Many implementations make that combination work through
model-family-specific subclasses and local assumptions about datasets or launch
environments.

LLaVA-Anything needs a smaller and more reusable foundation. The project should
let users compose compatible Transformers language and vision models, save the
result as a Hugging Face-style artifact, and load it through standard Auto APIs
whenever possible.

## Options Considered

| Option | Pros | Cons | When Valid |
| --- | --- | --- | --- |
| Maintain class-per-LLM wrappers | Fast when supporting one known model family | Does not scale well to arbitrary LLMs | A narrow reproduction-focused fork |
| Wrap an existing LLaVA implementation directly | Low initial implementation cost | Harder to expose generic composition and clean configs | Reproducing existing checkpoints |
| Build a Hugging Face-native composition layer | Clear component boundaries and reusable artifacts | Requires careful processor and generation integration | A general framework for new VLMs |

## Decision

Use a Hugging Face-native compositional architecture:

- `LlavaAnythingConfig` stores nested text and vision configs.
- `LlavaAnythingForConditionalGeneration` composes an
  `AutoModelForCausalLM`, an `AutoModel` vision tower, and a multimodal
  projector.
- `LlavaAnythingProcessor` wraps a tokenizer and image processor and expands
  image placeholders into the visual token layout expected by the model.
- YAML compiles into Hugging Face config, processor, and optional model
  artifacts.
- `trust_remote_code` is explicit in YAML and loading paths.

## Rationale

This design keeps the public contract close to Transformers conventions while
allowing the project to support more language models and vision towers over
time. It also makes trained checkpoints easier to share because downstream code
does not need to reconstruct the original YAML recipe.

## Trade-Offs

- Some base models may still need compatibility fixes when their generation or
  tokenizer behavior differs from common causal LM conventions.
- Evaluation frameworks with custom registries still need thin adapters.
- Any-resolution image packing adds complexity to processor behavior, so it
  needs strong tests and clear configuration docs.

## Consequences

- Positive: new compatible LLMs and vision towers can share one VLM wrapper.
- Positive: saved checkpoints can be consumed through Hugging Face Auto APIs.
- Positive: training, inference, and evaluation docs can describe one common
  artifact format.
- Negative: unsupported model families may fail later than a hand-written
  model-specific wrapper would.
- Mitigation: keep errors clear, add compatibility tests, and add small adapters
  only when a real model requires them.

## Revisit Trigger

Reconsider this decision if Transformers provides a stable generic multimodal
composition API that replaces this wrapper, or if supporting important model
families requires deep model-specific behavior in the core architecture.
