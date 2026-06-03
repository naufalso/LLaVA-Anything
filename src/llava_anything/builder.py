"""YAML builder utilities for LLaVa-Anything."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

from .configuration_llava_anything import LlavaAnythingConfig
from .modeling_llava_anything import LlavaAnythingForConditionalGeneration
from .processing_llava_anything import LlavaAnythingProcessor


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require its top-level value to be a mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a named YAML section, defaulting to an empty mapping."""

    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _extract_vision_config(config: Any) -> Any:
    """Return the vision-only config for CLIP/SigLIP aggregate configs."""

    if hasattr(config, "vision_config"):
        return config.vision_config
    return config


def config_from_yaml_dict(data: dict[str, Any]) -> LlavaAnythingConfig:
    """Construct a model config from a parsed LLaVa-Anything YAML mapping."""

    model_section = _section(data, "model")
    text_section = _section(data, "text_model")
    vision_section = _section(data, "vision_model")
    processor_section = _section(vision_section, "image_processor")
    image_section = _section(data, "image")
    anyres_section = _section(image_section, "anyres")

    text_name = text_section.get("name_or_path")
    vision_name = vision_section.get("name_or_path")
    if not text_name:
        raise ValueError("text_model.name_or_path is required.")
    if not vision_name:
        raise ValueError("vision_model.name_or_path is required.")

    text_trust_remote_code = bool(text_section.get("trust_remote_code", model_section.get("trust_remote_code", False)))
    vision_trust_remote_code = bool(vision_section.get("trust_remote_code", model_section.get("trust_remote_code", False)))
    trust_remote_code = bool(model_section.get("trust_remote_code", text_trust_remote_code or vision_trust_remote_code))

    text_config = AutoConfig.from_pretrained(text_name, trust_remote_code=text_trust_remote_code)
    vision_config = _extract_vision_config(
        AutoConfig.from_pretrained(vision_name, trust_remote_code=vision_trust_remote_code)
    )

    image_mode = image_section.get("mode", model_section.get("image_mode", "fixed"))
    if anyres_section.get("enabled") is True:
        image_mode = "anyres"
    image_grid_pinpoints = anyres_section.get("grid_pinpoints", model_section.get("image_grid_pinpoints"))

    image_seq_length = model_section.get("image_seq_length")
    if image_seq_length is None:
        patch_size = processor_section.get("patch_size") or getattr(vision_config, "patch_size", None)
        image_size = getattr(vision_config, "image_size", None)
        if patch_size is not None and image_size is not None:
            image_seq_length = (int(image_size) // int(patch_size)) ** 2
            if model_section.get("vision_feature_select_strategy", "default") == "full":
                image_seq_length += int(processor_section.get("num_additional_image_tokens", 0))

    config = LlavaAnythingConfig.from_text_vision_configs(
        text_config=text_config,
        vision_config=vision_config,
        image_token=model_section.get("image_token", "<image>"),
        image_token_index=int(model_section.get("image_token_index", 32000)),
        projector_type=model_section.get("projector_type", "mlp2x_gelu"),
        projector_hidden_act=model_section.get("projector_hidden_act", "gelu"),
        vision_feature_layer=model_section.get("vision_feature_layer", -2),
        vision_feature_select_strategy=model_section.get("vision_feature_select_strategy", "default"),
        image_seq_length=image_seq_length,
        num_additional_image_tokens=int(processor_section.get("num_additional_image_tokens", 1)),
        image_mode=image_mode,
        image_grid_pinpoints=image_grid_pinpoints,
        text_model_name_or_path=text_name,
        vision_model_name_or_path=vision_name,
        trust_remote_code=trust_remote_code,
        text_trust_remote_code=text_trust_remote_code,
        vision_trust_remote_code=vision_trust_remote_code,
    )
    config.architectures = ["LlavaAnythingForConditionalGeneration"]
    return config


def config_from_yaml(path: str | Path) -> LlavaAnythingConfig:
    """Load YAML from disk and convert it into a LLaVa-Anything config."""

    return config_from_yaml_dict(load_yaml(path))


def _token_in_vocab(tokenizer: Any, token: str) -> bool:
    """Return whether a tokenizer already knows a token."""

    try:
        return token in tokenizer.get_vocab()
    except AttributeError:
        return tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id


def processor_from_yaml_dict(data: dict[str, Any], config: LlavaAnythingConfig) -> LlavaAnythingProcessor:
    """Build a processor and synchronize tokenizer image-token metadata with the config."""

    text_section = _section(data, "text_model")
    vision_section = _section(data, "vision_model")
    tokenizer_section = _section(text_section, "tokenizer")
    image_processor_section = _section(vision_section, "image_processor")

    tokenizer_kwargs = {"trust_remote_code": config.text_trust_remote_code}
    if tokenizer_section.get("use_fast") is not None:
        tokenizer_kwargs["use_fast"] = tokenizer_section["use_fast"]
    tokenizer = AutoTokenizer.from_pretrained(text_section["name_or_path"], **tokenizer_kwargs)
    if tokenizer_section.get("padding_side") is not None:
        tokenizer.padding_side = tokenizer_section["padding_side"]
    if tokenizer_section.get("model_max_length") is not None:
        tokenizer.model_max_length = int(tokenizer_section["model_max_length"])

    if not _token_in_vocab(tokenizer, config.image_token):
        additional_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if config.image_token not in additional_special_tokens:
            additional_special_tokens.append(config.image_token)
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    config.image_token_index = int(tokenizer.convert_tokens_to_ids(config.image_token))
    config.vocab_size = int(getattr(config.text_config, "vocab_size", len(tokenizer)) or len(tokenizer))

    image_processor = AutoImageProcessor.from_pretrained(
        vision_section["name_or_path"],
        trust_remote_code=config.vision_trust_remote_code,
    )

    chat_template = tokenizer_section.get("chat_template", getattr(tokenizer, "chat_template", None))
    return LlavaAnythingProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_token=config.image_token,
        image_seq_length=config.image_seq_length,
        patch_size=image_processor_section.get("patch_size", getattr(config.vision_config, "patch_size", None)),
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        num_additional_image_tokens=config.num_additional_image_tokens,
        image_mode=getattr(config, "image_mode", "fixed"),
        image_grid_pinpoints=getattr(config, "image_grid_pinpoints", None),
        chat_template=chat_template,
    )


def processor_from_yaml(path: str | Path, config: LlavaAnythingConfig | None = None) -> LlavaAnythingProcessor:
    """Load YAML from disk and build a matching LLaVa-Anything processor."""

    data = load_yaml(path)
    config = config or config_from_yaml_dict(data)
    return processor_from_yaml_dict(data, config)


def model_from_yaml_dict(
    data: dict[str, Any],
    config: LlavaAnythingConfig,
    load_pretrained_components: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> LlavaAnythingForConditionalGeneration:
    """Build a model from YAML data, optionally loading pretrained component weights."""

    if not load_pretrained_components:
        return LlavaAnythingForConditionalGeneration(config)

    model_kwargs = dict(model_kwargs or {})
    text_model_kwargs = dict(model_kwargs.pop("text_model_kwargs", {}))
    vision_model_kwargs = dict(model_kwargs.pop("vision_model_kwargs", {}))
    if model_kwargs:
        text_model_kwargs.update(model_kwargs)
        vision_model_kwargs.update(model_kwargs)

    return LlavaAnythingForConditionalGeneration.from_pretrained_components(
        text_model_name_or_path=config.text_model_name_or_path or _section(data, "text_model")["name_or_path"],
        vision_model_name_or_path=config.vision_model_name_or_path or _section(data, "vision_model")["name_or_path"],
        config=config,
        trust_remote_code=config.trust_remote_code,
        text_model_kwargs=text_model_kwargs,
        vision_model_kwargs=vision_model_kwargs,
    )


def save_from_yaml(
    yaml_path: str | Path,
    output_dir: str | Path,
    load_pretrained_components: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> None:
    """Materialize config, processor, and optionally model weights into an output directory."""

    data = load_yaml(yaml_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = config_from_yaml_dict(data)
    processor = processor_from_yaml_dict(data, config)
    config.save_pretrained(output)
    processor.save_pretrained(output)

    if load_pretrained_components:
        model = model_from_yaml_dict(
            data,
            config=config,
            load_pretrained_components=True,
            model_kwargs=model_kwargs,
        )
        tokenizer_vocab_size = len(processor.tokenizer)
        embedding_vocab_size = model.get_input_embeddings().num_embeddings
        if tokenizer_vocab_size > embedding_vocab_size:
            try:
                model.resize_token_embeddings(tokenizer_vocab_size, mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(tokenizer_vocab_size)
            resized_vocab_size = model.get_input_embeddings().num_embeddings
            model.config.text_config.vocab_size = resized_vocab_size
            model.config.vocab_size = resized_vocab_size
        model.save_pretrained(output)


def main() -> None:
    """CLI entry point for building LLaVa-Anything artifacts from YAML."""

    parser = argparse.ArgumentParser(description="Build a LLaVa-Anything config/processor from YAML.")
    parser.add_argument("yaml_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--load-pretrained-components",
        action="store_true",
        help="Load and save the base LLM and vision weights. This can require substantial memory.",
    )
    args = parser.parse_args()
    save_from_yaml(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        load_pretrained_components=args.load_pretrained_components,
    )


__all__ = [
    "config_from_yaml",
    "config_from_yaml_dict",
    "load_yaml",
    "model_from_yaml_dict",
    "processor_from_yaml",
    "processor_from_yaml_dict",
    "save_from_yaml",
]
