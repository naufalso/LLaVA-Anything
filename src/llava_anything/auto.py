"""Auto-class registration helpers."""

from __future__ import annotations

from transformers import AutoConfig, AutoProcessor

from .configuration_llava_anything import LlavaAnythingConfig
from .modeling_llava_anything import LlavaAnythingForConditionalGeneration
from .processing_llava_anything import LlavaAnythingProcessor


def _register(registry, *args) -> None:
    """Register an Auto class mapping while tolerating older Transformers APIs."""

    try:
        registry.register(*args, exist_ok=True)
    except TypeError:
        try:
            registry.register(*args)
        except ValueError:
            pass


def register_auto_classes() -> None:
    """Register LLaVa-Anything config, model, and processor with Transformers Auto classes."""

    _register(AutoConfig, LlavaAnythingConfig.model_type, LlavaAnythingConfig)

    try:
        from transformers import AutoModelForImageTextToText

        _register(
            AutoModelForImageTextToText,
            LlavaAnythingConfig,
            LlavaAnythingForConditionalGeneration,
        )
    except (ImportError, AttributeError):
        try:
            from transformers import AutoModelForVision2Seq

            _register(
                AutoModelForVision2Seq,
                LlavaAnythingConfig,
                LlavaAnythingForConditionalGeneration,
            )
        except (ImportError, AttributeError):
            pass

    try:
        _register(AutoProcessor, LlavaAnythingConfig, LlavaAnythingProcessor)
    except (AttributeError, TypeError):
        pass


__all__ = ["register_auto_classes"]
