"""PyTorch model for LLaVa-Anything."""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.image_processing_utils import select_best_resolution
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


def _default_return_dict_from_config(config: Any) -> bool:
    """Read the return-dict preference from configs across Transformers versions."""

    missing = object()
    return_dict = getattr(config, "return_dict", missing)
    if return_dict is not missing:
        return bool(return_dict)
    return bool(getattr(config, "use_return_dict"))


def _is_expected_ignored_vision_key(key: str) -> bool:
    """Return whether an unused vision checkpoint key is expected for CLIP-style models."""

    return key.startswith("text_model.") or key in {
        "logit_scale",
        "text_projection.weight",
        "visual_projection.weight",
    }


def _format_key_examples(keys: list[str], *, limit: int = 6) -> list[str]:
    """Format a short bullet list of checkpoint key examples."""

    examples = keys[:limit]
    lines = [f"  - {key}" for key in examples]
    remaining = len(keys) - len(examples)
    if remaining > 0:
        lines.append(f"  - ... and {remaining} more")
    return lines


def _format_component_load_report(
    component_name: str,
    model_name_or_path: str,
    loading_info: dict[str, Any] | None,
) -> str | None:
    """Build a human-readable report for partial component checkpoint loading."""

    if not loading_info:
        return None

    missing_keys = list(loading_info.get("missing_keys", []) or [])
    unexpected_keys = list(loading_info.get("unexpected_keys", []) or [])
    mismatched_keys = list(loading_info.get("mismatched_keys", []) or [])
    error_msgs = list(loading_info.get("error_msgs", []) or [])
    if not (missing_keys or unexpected_keys or mismatched_keys or error_msgs):
        return None

    lines = [f"[llava-anything] {component_name} load summary from: {model_name_or_path}"]
    if unexpected_keys:
        expected_vision_only = all(_is_expected_ignored_vision_key(str(key)) for key in unexpected_keys)
        lines.append(f"ignored checkpoint weights: {len(unexpected_keys)}")
        if expected_vision_only:
            lines.append(
                "  These are expected when loading only the vision encoder from a CLIP-style checkpoint; "
                "the CLIP text tower and contrastive heads are not used."
            )
        else:
            lines.append("  These weights exist in the checkpoint but are not used by this component.")
        lines.extend(_format_key_examples([str(key) for key in unexpected_keys]))
    if missing_keys:
        lines.append(f"Missing model weights initialized by this run: {len(missing_keys)}")
        lines.extend(_format_key_examples([str(key) for key in missing_keys]))
    if mismatched_keys:
        lines.append(f"Shape-mismatched weights skipped: {len(mismatched_keys)}")
        lines.extend(_format_key_examples([str(key) for key in mismatched_keys]))
    if error_msgs:
        lines.append("Loader messages:")
        lines.extend(_format_key_examples([str(message) for message in error_msgs]))
    return "\n".join(lines)


def _as_image_size(image_size: Any) -> list[int]:
    """Normalize an image-size object to a two-integer ``[height, width]`` list."""

    if isinstance(image_size, (list, tuple)):
        return [int(image_size[0]), int(image_size[1])]
    if isinstance(image_size, torch.Tensor):
        return [int(value) for value in image_size.detach().cpu().tolist()]
    if hasattr(image_size, "tolist"):
        return [int(value) for value in image_size.tolist()]
    raise TypeError(f"Unsupported image_size type: {type(image_size)!r}")


def get_anyres_image_grid_shape(image_size: Any, grid_pinpoints: list[list[int]], patch_size: int) -> tuple[int, int]:
    """Return the selected any-resolution grid shape in patch units."""

    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of [height, width] pairs.")
    best_height, best_width = select_best_resolution(_as_image_size(image_size), grid_pinpoints)
    return (best_height + patch_size - 1) // patch_size, (best_width + patch_size - 1) // patch_size


def image_size_to_num_patches(image_size: Any, grid_pinpoints: list[list[int]], patch_size: int) -> int:
    """Return the number of any-resolution image crops, including the base image."""

    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of [height, width] pairs.")
    best_height, best_width = select_best_resolution(_as_image_size(image_size), grid_pinpoints)
    return ((best_height + patch_size - 1) // patch_size) * ((best_width + patch_size - 1) // patch_size) + 1


def unpad_image(tensor: torch.Tensor, original_size: Any) -> torch.Tensor:
    """Remove aspect-ratio padding from a feature map using the original image size."""

    original_height, original_width = _as_image_size(original_size)
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]

    scale_factor = current_height / original_height
    new_width = int(round(original_width * scale_factor, 7))
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]


class IdentityProjector(nn.Module):
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Return image features unchanged."""

        return image_features


class LlavaAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaAnythingConfig) -> None:
        """Create the configured multimodal projector between vision and text dimensions."""

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
        """Project vision features into the language-model embedding space."""

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

    _legacy_llava_next_key_mapping = {
        r"^model\.vision_tower\.vision_tower\.vision_model\.(.*)$": r"vision_tower.\1",
        r"^model\.mm_projector\.(.*)$": r"multi_modal_projector.layers.\1",
        r"^model\.image_newline$": "image_newline",
        r"^lm_head\.weight$": "language_model.lm_head.weight",
        r"^model\.(?!(vision_tower|mm_projector|image_newline)$)(.*)$": r"language_model.model.\2",
    }

    @staticmethod
    def _remap_legacy_llava_next_key(key: str) -> str:
        """Map legacy LLaVA-NeXT checkpoint keys to this model's module names."""

        if key.startswith("model.vision_tower.vision_tower.vision_model."):
            return "vision_tower." + key.removeprefix("model.vision_tower.vision_tower.vision_model.")
        if key.startswith("model.mm_projector."):
            return "multi_modal_projector.layers." + key.removeprefix("model.mm_projector.")
        if key == "model.image_newline":
            return "image_newline"
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
        if key.startswith("model."):
            return "language_model." + key
        return key

    @classmethod
    def _fix_state_dict_key_on_load(cls, key: str) -> tuple[str, bool]:
        """Return a remapped state-dict key and whether it changed."""

        remapped_key = cls._remap_legacy_llava_next_key(key)
        return remapped_key, remapped_key != key

    @classmethod
    def _remap_legacy_llava_next_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap every legacy state-dict key while preserving loader metadata."""

        remapped = OrderedDict((cls._remap_legacy_llava_next_key(key), value) for key, value in state_dict.items())
        if hasattr(state_dict, "_metadata"):
            remapped._metadata = state_dict._metadata  # type: ignore[attr-defined]
        return remapped

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Load weights, automatically remapping supported legacy LLaVA-NeXT keys."""

        if any(
            key.startswith("model.mm_projector.")
            or key.startswith("model.vision_tower.vision_tower.")
            or key == "model.image_newline"
            for key in state_dict
        ):
            state_dict = self._remap_legacy_llava_next_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _checkpoint_config_flag(pretrained_model_name_or_path: str | Path) -> bool:
        """Check whether a checkpoint config declares legacy key compatibility."""

        try:
            config_path = Path(pretrained_model_name_or_path) / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (TypeError, OSError, ValueError):
            return False
        return bool(config.get("legacy_llava_next_checkpoint", False))

    @staticmethod
    def _checkpoint_has_legacy_llava_next_keys(pretrained_model_name_or_path: str | Path) -> bool:
        """Inspect a safetensors index for legacy LLaVA-NeXT key patterns."""

        try:
            model_path = Path(pretrained_model_name_or_path)
        except TypeError:
            return False
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            return False
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError):
            return False
        return any(
            key.startswith("model.mm_projector.")
            or key.startswith("model.vision_tower.vision_tower.vision_model.")
            or key == "model.image_newline"
            for key in weight_map
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path | None, *model_args: Any, **kwargs: Any):
        """Load a checkpoint, installing key remapping when legacy weights are detected."""

        if (
            pretrained_model_name_or_path is not None
            and kwargs.get("key_mapping") is None
            and (
                cls._checkpoint_config_flag(pretrained_model_name_or_path)
                or cls._checkpoint_has_legacy_llava_next_keys(pretrained_model_name_or_path)
            )
        ):
            kwargs["key_mapping"] = dict(cls._legacy_llava_next_key_mapping)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(
        self,
        config: LlavaAnythingConfig,
        language_model: PreTrainedModel | None = None,
        vision_tower: PreTrainedModel | None = None,
        init_weights: bool = True,
    ) -> None:
        """Initialize the vision tower, language model, projector, and newline embedding."""

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

    @property
    def device(self) -> torch.device:
        """Return the device of the model input embeddings."""

        return self.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model input embeddings."""

        return self.get_input_embeddings().weight.dtype

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
        """Assemble a LLaVa-Anything model from separate pretrained text and vision models."""

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
        embedding_vocab_size = language_model.get_input_embeddings().num_embeddings
        required_vocab_size = max(int(config.image_token_index) + 1, int(getattr(config, "vocab_size", 0) or 0))
        if required_vocab_size > embedding_vocab_size:
            try:
                language_model.resize_token_embeddings(required_vocab_size, mean_resizing=False)
            except TypeError:
                language_model.resize_token_embeddings(required_vocab_size)
            resized_vocab_size = language_model.get_input_embeddings().num_embeddings
            config.text_config.vocab_size = resized_vocab_size
            config.vocab_size = resized_vocab_size
        vision_model_kwargs.setdefault("output_loading_info", True)
        loaded_vision_tower = AutoModel.from_pretrained(
            vision_model_name_or_path,
            config=config.vision_config,
            trust_remote_code=config.vision_trust_remote_code,
            **vision_model_kwargs,
        )
        if isinstance(loaded_vision_tower, tuple):
            vision_tower, vision_loading_info = loaded_vision_tower
            load_report = _format_component_load_report(
                "vision tower",
                vision_model_name_or_path,
                vision_loading_info,
            )
            if load_report is not None:
                print(load_report)
        else:
            vision_tower = loaded_vision_tower
        return cls(config, language_model=language_model, vision_tower=vision_tower, init_weights=False)

    def get_input_embeddings(self) -> nn.Module:
        """Return the language model's input embedding module."""

        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the language model's input embedding module."""

        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module | None:
        """Return the language model's output embedding module, when available."""

        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Replace the language model's output embedding module."""

        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        """Return the decoder module used by the wrapped language model."""

        if hasattr(self.language_model, "get_decoder"):
            return self.language_model.get_decoder()
        return self.language_model

    def set_decoder(self, decoder: nn.Module) -> None:
        """Replace the wrapped language model decoder when the backend supports it."""

        if hasattr(self.language_model, "set_decoder"):
            self.language_model.set_decoder(decoder)
            return
        raise AttributeError(f"{self.language_model.__class__.__name__} does not expose set_decoder().")

    def _select_vision_features(self, vision_outputs: Any) -> torch.Tensor:
        """Select and optionally concatenate hidden states from configured vision layers."""

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

    def _project_vision_outputs(self, pixel_values: torch.FloatTensor, **kwargs: Any) -> torch.Tensor:
        """Run the vision tower and projector for a batch of image tensors."""

        vision_parameter = next(self.vision_tower.parameters())
        vision_outputs = self.vision_tower(
            pixel_values.to(device=vision_parameter.device, dtype=vision_parameter.dtype),
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        selected = self._select_vision_features(vision_outputs)
        projector_parameter = next(iter(self.multi_modal_projector.parameters()), None)
        if projector_parameter is not None and (
            projector_parameter.device != selected.device or projector_parameter.dtype != selected.dtype
        ):
            self.multi_modal_projector.to(device=selected.device, dtype=selected.dtype)
        return self.multi_modal_projector(selected)

    def pack_image_features(
        self,
        image_features: tuple[torch.Tensor, ...],
        image_sizes: torch.LongTensor,
    ) -> torch.Tensor:
        """Pack any-resolution crop features into a single sequence aligned to image tokens."""

        if self.config.image_grid_pinpoints is None:
            raise ValueError("image_grid_pinpoints is required when image_mode='anyres'.")

        new_image_features = []
        image_size = int(self.config.vision_config.image_size)
        patch_size = int(self.config.vision_config.patch_size)
        patch_grid_height = patch_grid_width = image_size // patch_size

        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                highres_feature = image_feature[1:]
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    image_size,
                )
                if num_patch_height * num_patch_width != highres_feature.shape[0]:
                    raise ValueError(
                        "Image feature grid does not match the processed patch count. "
                        f"grid={num_patch_height}x{num_patch_width}, features={tuple(highres_feature.shape)}."
                    )
                expected_patch_tokens = patch_grid_height * patch_grid_width
                if highres_feature.shape[1] != expected_patch_tokens:
                    raise ValueError(
                        "Image feature shape does not match the any-resolution grid. "
                        f"features={tuple(highres_feature.shape)}, expected_patch_tokens={expected_patch_tokens}."
                    )
                highres_feature = highres_feature.view(
                    num_patch_height,
                    num_patch_width,
                    patch_grid_height,
                    patch_grid_width,
                    -1,
                )
                highres_feature = highres_feature.permute(4, 0, 2, 1, 3).contiguous()
                highres_feature = highres_feature.flatten(1, 2).flatten(2, 3)
                highres_feature = unpad_image(highres_feature, image_sizes[image_idx])
                highres_feature = torch.cat(
                    (
                        highres_feature,
                        self.image_newline[:, None, None]
                        .expand(*highres_feature.shape[:-1], 1)
                        .to(highres_feature.device, highres_feature.dtype),
                    ),
                    dim=-1,
                )
                highres_feature = highres_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, highres_feature), dim=0)
            else:
                image_feature = image_feature[0]
                image_feature = torch.cat((image_feature, self.image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)

        return torch.cat(new_image_features, dim=0).to(device=self.device, dtype=self.dtype)

    def _get_anyres_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.LongTensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract and pack image features for any-resolution pixel-value tensors."""

        if image_sizes is None:
            raise ValueError("image_sizes is required when image_mode='anyres'.")
        if self.config.image_grid_pinpoints is None:
            raise ValueError("image_grid_pinpoints is required when image_mode='anyres'.")

        image_size = int(self.config.vision_config.image_size)
        image_num_patches = [
            image_size_to_num_patches(
                image_size=image_size_value,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=image_size,
            )
            for image_size_value in image_sizes
        ]
        if pixel_values.dim() == 5:
            pixel_values = torch.cat(
                [
                    image_pixel_values[:num_patches]
                    for image_pixel_values, num_patches in zip(pixel_values, image_num_patches)
                ],
                dim=0,
            )
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be 4D or 5D, got shape {tuple(pixel_values.shape)}")

        projected = self._project_vision_outputs(pixel_values, **kwargs)
        image_features = torch.split(projected, image_num_patches, dim=0)
        return self.pack_image_features(image_features, image_sizes)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract projected image features for fixed or any-resolution inputs."""

        if getattr(self.config, "image_mode", "fixed") == "anyres":
            return self._get_anyres_image_features(pixel_values, image_sizes, **kwargs)

        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be 4D or 5D, got shape {tuple(pixel_values.shape)}")

        image_features = self._project_vision_outputs(pixel_values, **kwargs)
        return image_features.to(device=self.device, dtype=self.dtype)

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Replace image-token embeddings with projected vision features."""

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
        """Run multimodal conditional generation with optional image-feature injection."""

        return_dict = return_dict if return_dict is not None else _default_return_dict_from_config(self.config)
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
        """Prepare decoder inputs while keeping image tensors only when needed."""

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
