"""PyTorch model for LLaVa-Anything."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

try:
    from transformers.generation import GenerationMixin
except ImportError:  # pragma: no cover - older Transformers fallback
    from transformers.generation.utils import GenerationMixin

from .configuration_llava_anything import LlavaAnythingConfig


@dataclass
class LlavaAnythingCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class IdentityProjector(nn.Module):
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return image_features


class LlavaAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaAnythingConfig) -> None:
        super().__init__()
        vision_hidden_size = getattr(config.vision_config, "hidden_size", None)
        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        if vision_hidden_size is None or text_hidden_size is None:
            raise ValueError("Both vision_config.hidden_size and text_config.hidden_size are required.")

        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        in_features = vision_hidden_size * num_feature_layers
        projector_type = config.projector_type

        if projector_type == "identity":
            if in_features != text_hidden_size:
                raise ValueError("The identity projector requires matching vision/text hidden sizes.")
            self.layers = IdentityProjector()
        elif projector_type == "linear":
            self.layers = nn.Linear(in_features, text_hidden_size)
        else:
            mlp_match = re.fullmatch(r"mlp(\d+)x_gelu", projector_type)
            if mlp_match is None:
                raise ValueError(f"Unsupported projector_type: {projector_type}")
            depth = int(mlp_match.group(1))
            layers: list[nn.Module] = [nn.Linear(in_features, text_hidden_size)]
            for _ in range(1, depth):
                layers.append(nn.GELU() if config.projector_hidden_act == "gelu" else ACT2FN[config.projector_hidden_act])
                layers.append(nn.Linear(text_hidden_size, text_hidden_size))
            self.layers = nn.Sequential(*layers)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.layers(image_features)


class LlavaAnythingForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = LlavaAnythingConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _skip_keys_device_placement = "past_key_values"

    def __init__(
        self,
        config: LlavaAnythingConfig,
        language_model: PreTrainedModel | None = None,
        vision_tower: PreTrainedModel | None = None,
        init_weights: bool = True,
    ) -> None:
        super().__init__(config)
        self.vision_tower = vision_tower if vision_tower is not None else AutoModel.from_config(config.vision_config)
        self.language_model = (
            language_model if language_model is not None else AutoModelForCausalLM.from_config(config.text_config)
        )
        self.multi_modal_projector = LlavaAnythingMultiModalProjector(config)

        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size) * embed_std)

        if init_weights:
            self.post_init()
        else:
            self.multi_modal_projector.apply(self._init_weights)
            nn.init.normal_(self.image_newline, mean=0.0, std=embed_std)

    @classmethod
    def from_pretrained_components(
        cls,
        text_model_name_or_path: str,
        vision_model_name_or_path: str,
        config: LlavaAnythingConfig | None = None,
        trust_remote_code: bool = False,
        text_model_kwargs: dict[str, Any] | None = None,
        vision_model_kwargs: dict[str, Any] | None = None,
    ) -> "LlavaAnythingForConditionalGeneration":
        text_model_kwargs = dict(text_model_kwargs or {})
        vision_model_kwargs = dict(vision_model_kwargs or {})
        if config is None:
            text_config = text_model_kwargs.pop("config", None)
            vision_config = vision_model_kwargs.pop("config", None)
            if text_config is None:
                from transformers import AutoConfig

                text_config = AutoConfig.from_pretrained(text_model_name_or_path, trust_remote_code=trust_remote_code)
            if vision_config is None:
                from transformers import AutoConfig

                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path, trust_remote_code=trust_remote_code)
            config = LlavaAnythingConfig.from_text_vision_configs(
                text_config=text_config,
                vision_config=vision_config,
                text_model_name_or_path=text_model_name_or_path,
                vision_model_name_or_path=vision_model_name_or_path,
                trust_remote_code=trust_remote_code,
            )

        language_model = AutoModelForCausalLM.from_pretrained(
            text_model_name_or_path,
            config=config.text_config,
            trust_remote_code=config.text_trust_remote_code,
            **text_model_kwargs,
        )
        vision_tower = AutoModel.from_pretrained(
            vision_model_name_or_path,
            config=config.vision_config,
            trust_remote_code=config.vision_trust_remote_code,
            **vision_model_kwargs,
        )
        return cls(config, language_model=language_model, vision_tower=vision_tower, init_weights=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module | None:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        if hasattr(self.language_model, "get_decoder"):
            return self.language_model.get_decoder()
        return self.language_model

    def set_decoder(self, decoder: nn.Module) -> None:
        if hasattr(self.language_model, "set_decoder"):
            self.language_model.set_decoder(decoder)
            return
        raise AttributeError(f"{self.language_model.__class__.__name__} does not expose set_decoder().")

    def _select_vision_features(self, vision_outputs: Any) -> torch.Tensor:
        hidden_states = vision_outputs.hidden_states
        if hidden_states is None:
            raise ValueError("The vision tower must return hidden_states. Pass output_hidden_states=True.")

        layer = self.config.vision_feature_layer
        if isinstance(layer, int):
            selected = hidden_states[layer]
        else:
            selected = torch.cat([hidden_states[idx] for idx in layer], dim=-1)

        if self.config.vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        elif self.config.vision_feature_select_strategy != "full":
            raise ValueError(
                "vision_feature_select_strategy must be either 'default' or 'full', "
                f"got {self.config.vision_feature_select_strategy!r}."
            )
        return selected

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del image_sizes
        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be 4D or 5D, got shape {tuple(pixel_values.shape)}")

        vision_outputs = self.vision_tower(
            pixel_values.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        selected = self._select_vision_features(vision_outputs)
        image_features = self.multi_modal_projector(selected)
        return image_features.to(dtype=self.get_input_embeddings().weight.dtype)

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids is None:
            image_token = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token).all(dim=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index

        image_features = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.device, inputs_embeds.dtype)
        num_image_tokens = int(special_image_mask.sum().item())
        if num_image_tokens != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={num_image_tokens}, features={image_features.shape[0]}. "
                "Make sure the processor expands the image token using the same image_seq_length as the model."
            )

        mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask, image_features)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | LlavaAnythingCausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if kwargs.get("logits_to_keep", 0) is None:
            kwargs.pop("logits_to_keep")
        image_hidden_states = None

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            image_hidden_states = self.get_image_features(pixel_values=pixel_values, image_sizes=image_sizes)
            inputs_embeds = self._merge_input_ids_with_image_features(input_ids, inputs_embeds, image_hidden_states)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        if not return_dict:
            return outputs.to_tuple()

        return LlavaAnythingCausalLMOutputWithPast(
            loss=outputs.loss if hasattr(outputs, "loss") else None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        is_first_iteration: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        generation_kwargs = dict(kwargs)
        if logits_to_keep is not None:
            generation_kwargs["logits_to_keep"] = logits_to_keep
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            is_first_iteration=is_first_iteration,
            **generation_kwargs,
        )
        if is_first_iteration or past_key_values is None or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
        return model_inputs


__all__ = [
    "LlavaAnythingCausalLMOutputWithPast",
    "LlavaAnythingForConditionalGeneration",
    "LlavaAnythingMultiModalProjector",
]
