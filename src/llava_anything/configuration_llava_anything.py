"""Configuration for LLaVa-Anything."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class LlavaAnythingConfig(PretrainedConfig):
    """Configuration for a compositional LLaVA-style VLM.

    The config intentionally stores the text and vision configs as nested
    Transformers configs so model families can be selected by YAML rather than
    by adding a new Python subclass for every LLM.
    """

    model_type = "llava_anything"
    attribute_map = {"image_token_id": "image_token_index"}
    is_composition = True

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        image_token_index: int = 32000,
        image_token: str = "<image>",
        projector_type: str = "mlp2x_gelu",
        projector_hidden_act: str = "gelu",
        vision_feature_layer: int | list[int] = -2,
        vision_feature_select_strategy: str = "default",
        image_seq_length: int | None = None,
        num_additional_image_tokens: int = 1,
        text_model_name_or_path: str | None = None,
        vision_model_name_or_path: str | None = None,
        trust_remote_code: bool = False,
        text_trust_remote_code: bool | None = None,
        vision_trust_remote_code: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.text_config = self._coerce_config(text_config, default_model_type="llama")
        self.vision_config = self._coerce_config(vision_config, default_model_type="clip_vision_model")

        self.image_token_index = image_token_index
        self.image_token = image_token
        self.projector_type = projector_type
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_seq_length = image_seq_length
        self.num_additional_image_tokens = num_additional_image_tokens

        self.text_model_name_or_path = text_model_name_or_path
        self.vision_model_name_or_path = vision_model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.text_trust_remote_code = trust_remote_code if text_trust_remote_code is None else text_trust_remote_code
        self.vision_trust_remote_code = trust_remote_code if vision_trust_remote_code is None else vision_trust_remote_code

        self.vocab_size = getattr(self.text_config, "vocab_size", None)
        self.hidden_size = getattr(self.text_config, "hidden_size", None)

    @staticmethod
    def _coerce_config(
        config: dict[str, Any] | PretrainedConfig | None,
        default_model_type: str,
    ) -> PretrainedConfig:
        if isinstance(config, PretrainedConfig):
            return config
        if config is None:
            return AutoConfig.for_model(default_model_type)
        if not isinstance(config, dict):
            raise TypeError(f"Expected a dict or PretrainedConfig, got {type(config)!r}")

        config_dict = dict(config)
        model_type = config_dict.pop("model_type", default_model_type)
        try:
            return AutoConfig.for_model(model_type, **config_dict)
        except ValueError:
            config_dict["model_type"] = model_type
            return PretrainedConfig.from_dict(config_dict)

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        **kwargs: Any,
    ) -> "LlavaAnythingConfig":
        return cls(text_config=text_config, vision_config=vision_config, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        return output


__all__ = ["LlavaAnythingConfig"]

