"""OpenCLIP vision tower adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import CLIPImageProcessor
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

OPEN_CLIP_BACKENDS = {"open_clip", "open-clip"}
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
OPEN_CLIP_DEPENDENCY_ERROR = (
    "OpenCLIP vision towers require the optional 'open_clip_torch' package. "
    "Install it with `pip install 'llava-anything[openclip]'` or install "
    "`open_clip_torch` in the active environment."
)


class OpenCLIPVisionConfig(PretrainedConfig):
    """Configuration metadata needed to wrap an OpenCLIP vision tower."""

    model_type = "open_clip_vision_model"

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int | None = None,
        open_clip_model_name: str | None = None,
        open_clip_pretrained: str | None = None,
        open_clip_model_cfg: dict[str, Any] | None = None,
        open_clip_preprocess_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.open_clip_model_name = open_clip_model_name
        self.open_clip_pretrained = open_clip_pretrained
        self.open_clip_model_cfg = open_clip_model_cfg
        self.open_clip_preprocess_cfg = open_clip_preprocess_cfg or {}


def is_open_clip_vision_config(config: Any) -> bool:
    """Return whether a config describes an OpenCLIP vision tower."""

    return getattr(config, "model_type", None) == OpenCLIPVisionConfig.model_type


def is_open_clip_backend(value: Any) -> bool:
    """Return whether a YAML backend value selects OpenCLIP."""

    return str(value or "").lower() in OPEN_CLIP_BACKENDS


def normalize_open_clip_model_name(name_or_path: str | Path) -> str:
    """Normalize YAML model names into the format expected by OpenCLIP."""

    name = str(name_or_path)
    if name.startswith(("hf-hub:", "local-dir:")):
        return name
    if Path(name).exists():
        return f"local-dir:{name}"
    if "/" in name:
        return f"hf-hub:{name}"
    return name


def _open_clip_config_path(model_name: str) -> Path | None:
    """Return a local OpenCLIP config path when the model name points to one."""

    if model_name.startswith("local-dir:"):
        return Path(model_name.removeprefix("local-dir:")) / "open_clip_config.json"
    local_path = Path(model_name)
    if local_path.exists():
        return local_path / "open_clip_config.json"
    return None


def load_open_clip_config_dict(model_name: str) -> dict[str, Any]:
    """Load ``open_clip_config.json`` from a local directory or Hugging Face Hub repo."""

    config_path = _open_clip_config_path(model_name)
    if config_path is not None:
        return json.loads(config_path.read_text(encoding="utf-8"))

    if model_name.startswith("hf-hub:"):
        repo_id = model_name.removeprefix("hf-hub:")
    elif "/" in model_name:
        repo_id = model_name
    else:
        raise ValueError(
            "OpenCLIP backend needs either a local directory, an hf-hub:<repo> model name, "
            "or a Hugging Face repo id containing '/'."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - runtime dependency is already declared
        raise ImportError("Loading OpenCLIP configs from the Hub requires huggingface-hub.") from exc
    downloaded = hf_hub_download(repo_id=repo_id, filename="open_clip_config.json")
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def _as_int(value: Any, field_name: str) -> int:
    """Coerce scalar or square tuple/list OpenCLIP config values to an int."""

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"OpenCLIP vision_cfg.{field_name} cannot be empty.")
        return int(value[0])
    return int(value)


def config_from_open_clip_config(
    model_name_or_path: str | Path,
    *,
    pretrained: str | None = None,
) -> OpenCLIPVisionConfig:
    """Create a LLaVA-Anything vision config from OpenCLIP metadata."""

    model_name = normalize_open_clip_model_name(model_name_or_path)
    open_clip_config = load_open_clip_config_dict(model_name)
    model_cfg = dict(open_clip_config.get("model_cfg") or {})
    vision_cfg = dict(model_cfg.get("vision_cfg") or {})
    preprocess_cfg = dict(open_clip_config.get("preprocess_cfg") or {})
    if not vision_cfg:
        raise ValueError("open_clip_config.json must contain model_cfg.vision_cfg.")

    width = int(vision_cfg["width"])
    head_width = int(vision_cfg.get("head_width", 64))
    num_attention_heads = vision_cfg.get("heads")
    if num_attention_heads is None and head_width > 0:
        num_attention_heads = width // head_width

    return OpenCLIPVisionConfig(
        image_size=_as_int(vision_cfg["image_size"], "image_size"),
        patch_size=_as_int(vision_cfg["patch_size"], "patch_size"),
        hidden_size=width,
        num_hidden_layers=int(vision_cfg["layers"]),
        num_attention_heads=int(num_attention_heads) if num_attention_heads is not None else None,
        open_clip_model_name=model_name,
        open_clip_pretrained=pretrained,
        open_clip_model_cfg=model_cfg,
        open_clip_preprocess_cfg=preprocess_cfg,
    )


def image_processor_from_open_clip_config(
    config: OpenCLIPVisionConfig,
    overrides: dict[str, Any] | None = None,
) -> CLIPImageProcessor:
    """Build a Transformers image processor that mirrors OpenCLIP preprocessing."""

    overrides = dict(overrides or {})
    image_size = int(overrides.get("image_size", config.image_size))
    mean = overrides.get("image_mean", config.open_clip_preprocess_cfg.get("mean", OPENAI_CLIP_MEAN))
    std = overrides.get("image_std", config.open_clip_preprocess_cfg.get("std", OPENAI_CLIP_STD))
    return CLIPImageProcessor(
        size={"shortest_edge": image_size},
        crop_size={"height": image_size, "width": image_size},
        image_mean=list(mean),
        image_std=list(std),
    )


def _coerce_torch_dtype(dtype: Any) -> torch.dtype | None:
    """Normalize optional dtype values commonly passed through model kwargs."""

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        return getattr(torch, dtype_name)
    raise TypeError(f"Unsupported torch dtype value: {dtype!r}")


def _raise_missing_open_clip_dependency(exc: ImportError) -> None:
    """Raise the user-facing optional dependency error."""

    raise ImportError(OPEN_CLIP_DEPENDENCY_ERROR) from exc


class OpenCLIPVisionTower(nn.Module):
    """Expose an OpenCLIP visual tower with a Transformers-like output contract."""

    def __init__(self, config: OpenCLIPVisionConfig, visual: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.visual = visual if visual is not None else self._build_visual_from_config(config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        config: OpenCLIPVisionConfig,
        **kwargs: Any,
    ) -> "OpenCLIPVisionTower":
        """Load a pretrained OpenCLIP visual tower through the OpenCLIP factory."""

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on optional package
            _raise_missing_open_clip_dependency(exc)

        kwargs = dict(kwargs)
        torch_dtype = _coerce_torch_dtype(kwargs.pop("torch_dtype", None))
        kwargs.pop("output_loading_info", None)
        kwargs.pop("trust_remote_code", None)
        model_name = config.open_clip_model_name or normalize_open_clip_model_name(model_name_or_path)
        if config.open_clip_pretrained is not None:
            kwargs.setdefault("pretrained", config.open_clip_pretrained)
        clip_model = open_clip.create_model(model_name, **kwargs)
        tower = cls(config, visual=clip_model.visual)
        if torch_dtype is not None:
            tower = tower.to(dtype=torch_dtype)
        return tower

    @staticmethod
    def _build_visual_from_config(config: OpenCLIPVisionConfig) -> nn.Module:
        """Create an uninitialized OpenCLIP visual tower from saved model config."""

        if config.open_clip_model_cfg is None:
            raise ValueError("open_clip_model_cfg is required to initialize an OpenCLIP vision tower.")
        try:
            from open_clip.model import CLIP
        except ImportError as exc:  # pragma: no cover - depends on optional package
            _raise_missing_open_clip_dependency(exc)

        clip_model = CLIP(**config.open_clip_model_cfg)
        return clip_model.visual

    @staticmethod
    def _as_output_tuple(output: Any) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Extract intermediate patch and prefix tokens from OpenCLIP outputs."""

        if not isinstance(output, dict):
            raise TypeError("OpenCLIP forward_intermediates must return a mapping.")
        intermediates = list(output.get("image_intermediates") or [])
        prefix_tokens = output.get("image_intermediates_prefix")
        if prefix_tokens is not None:
            prefix_tokens = list(prefix_tokens)
        return intermediates, prefix_tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling | tuple[torch.Tensor, ...]:
        """Run the OpenCLIP visual tower and return hidden states for LLaVA selection."""

        if not hasattr(self.visual, "forward_intermediates"):
            raise TypeError("OpenCLIP visual tower must expose forward_intermediates().")

        output = self.visual.forward_intermediates(
            pixel_values,
            indices=None,
            stop_early=False,
            intermediates_only=True,
            output_fmt="NLC",
            output_extra_tokens=True,
            **kwargs,
        )
        intermediates, prefix_tokens = self._as_output_tuple(output)
        if not intermediates:
            raise ValueError("OpenCLIP visual tower did not return any image intermediates.")

        hidden_states = []
        for index, intermediate in enumerate(intermediates):
            if prefix_tokens is not None:
                intermediate = torch.cat([prefix_tokens[index], intermediate], dim=1)
            hidden_states.append(intermediate)

        last_hidden_state = hidden_states[-1]
        hidden_states_tuple = tuple(hidden_states) if output_hidden_states else None
        if not return_dict:
            output_tuple: tuple[torch.Tensor, ...] = (last_hidden_state,)
            if hidden_states_tuple is not None:
                output_tuple += (hidden_states_tuple,)
            return output_tuple

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=hidden_states_tuple,
        )


__all__ = [
    "OpenCLIPVisionConfig",
    "OpenCLIPVisionTower",
    "config_from_open_clip_config",
    "image_processor_from_open_clip_config",
    "is_open_clip_backend",
    "is_open_clip_vision_config",
    "load_open_clip_config_dict",
    "normalize_open_clip_model_name",
]
