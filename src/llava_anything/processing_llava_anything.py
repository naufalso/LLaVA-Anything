"""Processor for LLaVa-Anything."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.processing_utils import ProcessorMixin


class LlavaAnythingProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_token: str = "<image>",
        image_seq_length: int | None = None,
        patch_size: int | None = None,
        vision_feature_select_strategy: str = "default",
        num_additional_image_tokens: int = 1,
        image_mode: str = "fixed",
        image_grid_pinpoints: list[list[int]] | None = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        if image_mode not in {"fixed", "anyres"}:
            raise ValueError(f"image_mode must be 'fixed' or 'anyres', got {image_mode!r}.")
        self.image_token = image_token
        self.image_seq_length = image_seq_length
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_additional_image_tokens = num_additional_image_tokens
        self.image_mode = image_mode
        self.image_grid_pinpoints = image_grid_pinpoints
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        names = list(getattr(self.tokenizer, "model_input_names", ["input_ids", "attention_mask"]))
        if "pixel_values" not in names:
            names.append("pixel_values")
        if "image_sizes" not in names:
            names.append("image_sizes")
        return names

    def _processed_image_size(self) -> tuple[int, int]:
        size = getattr(self.image_processor, "crop_size", None) or getattr(self.image_processor, "size", None)
        if size is None:
            raise ValueError("image processor size is required for image-token expansion.")
        if isinstance(size, dict) or hasattr(size, "height"):
            height = getattr(size, "height", None) or size.get("height") or size.get("shortest_edge")
            width = getattr(size, "width", None) or size.get("width") or size.get("shortest_edge")
        else:
            height, width = size
        if height is None or width is None:
            raise ValueError("image processor size must include height and width.")
        return int(height), int(width)

    def _num_fixed_image_tokens(self) -> int:
        if self.image_seq_length is not None:
            return self.image_seq_length

        height, width = self._processed_image_size()
        if self.patch_size is None:
            raise ValueError("image_seq_length is required when patch_size is unknown.")
        tokens = (height // self.patch_size) * (width // self.patch_size)
        if self.vision_feature_select_strategy == "full":
            tokens += self.num_additional_image_tokens
        return tokens

    def _get_unpadded_features(
        self,
        height: int,
        width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
    ) -> tuple[int, int]:
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        return unpadded_features, newline_features

    def _num_anyres_image_tokens(self, image_size: Iterable[int]) -> int:
        if self.patch_size is None:
            raise ValueError("patch_size is required for any-resolution image-token expansion.")
        if self.image_grid_pinpoints is None:
            raise ValueError("image_grid_pinpoints is required when image_mode='anyres'.")

        orig_height, orig_width = [int(value) for value in image_size]
        height, width = self._processed_image_size()
        best_height, best_width = select_best_resolution([orig_height, orig_width], self.image_grid_pinpoints)
        scale_height = best_height // height
        scale_width = best_width // width
        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height,
            orig_width,
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        tokens = unpadded_features + newline_features + base_features
        if self.vision_feature_select_strategy == "default":
            tokens -= self.num_additional_image_tokens
        return tokens

    def _num_image_tokens(self, image_size: Iterable[int] | None = None) -> int:
        if self.image_mode == "anyres":
            if image_size is None:
                raise ValueError("image sizes are required to expand image tokens in any-resolution mode.")
            return self._num_anyres_image_tokens(image_size)
        return self._num_fixed_image_tokens()

    def _expand_image_tokens(self, text: str, image_sizes: Iterable[Iterable[int]] | None = None) -> str:
        placeholder = "<llava_anything_image_placeholder>"
        image_size_iter = iter(image_sizes or [])
        while self.image_token in text:
            image_size = next(image_size_iter, None)
            count = self._num_image_tokens(image_size)
            text = text.replace(self.image_token, placeholder * count, 1)
        return text.replace(placeholder, self.image_token)

    def _image_from_content_item(self, item: dict[str, Any]) -> Any | None:
        for key in ("image", "url", "path"):
            if key in item:
                return item[key]
        return None

    def _normalize_content_and_images(self, content: Any) -> tuple[str, list[Any]]:
        if isinstance(content, str):
            return content, []
        if not isinstance(content, list):
            return str(content), []

        parts: list[str] = []
        images: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                parts.append(self.image_token)
                image = self._image_from_content_item(item)
                if image is not None:
                    images.append(image)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part), images

    def _normalize_content(self, content: Any) -> str:
        text, _ = self._normalize_content_and_images(content)
        return text

    def _normalize_conversation_and_images(self, conversation: Any) -> tuple[Any, list[Any]]:
        if not isinstance(conversation, list):
            return conversation, []
        normalized = []
        images: list[Any] = []
        for message in conversation:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            copied = dict(message)
            content, message_images = self._normalize_content_and_images(copied.get("content", ""))
            copied["content"] = content
            images.extend(message_images)
            normalized.append(copied)
        return normalized, images

    def _normalize_conversation(self, conversation: Any) -> Any:
        normalized, _ = self._normalize_conversation_and_images(conversation)
        return normalized

    def _processor_kwargs_from_chat_template_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        processor_kwargs = kwargs.pop("processor_kwargs", None) or {}
        if not isinstance(processor_kwargs, dict):
            raise TypeError("processor_kwargs must be a mapping when provided.")
        return dict(processor_kwargs)

    def _tokenize_chat_template_output(
        self,
        rendered: str,
        images: list[Any],
        return_tensors: str | None,
        processor_kwargs: dict[str, Any],
    ) -> BatchFeature:
        image_inputs = images if images else None
        return self(images=image_inputs, text=rendered, return_tensors=return_tensors, **processor_kwargs)

    def apply_chat_template(self, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        tokenize = bool(kwargs.get("tokenize", False))
        return_dict = bool(kwargs.get("return_dict", False))
        return_tensors = kwargs.get("return_tensors")
        processor_kwargs = self._processor_kwargs_from_chat_template_kwargs(kwargs)
        conversation, images = self._normalize_conversation_and_images(conversation)

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            if tokenize and return_dict:
                template_kwargs = dict(kwargs)
                template_kwargs["tokenize"] = False
                template_kwargs.pop("return_tensors", None)
                template_kwargs.pop("return_dict", None)
                rendered = self.tokenizer.apply_chat_template(conversation, *args, **template_kwargs)
                return self._tokenize_chat_template_output(rendered, images, return_tensors, processor_kwargs)
            return self.tokenizer.apply_chat_template(conversation, *args, **kwargs)

        add_generation_prompt = kwargs.get("add_generation_prompt", False)
        lines = []
        for message in conversation:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("assistant:")
        rendered = "\n".join(lines)
        if tokenize:
            if return_dict:
                return self._tokenize_chat_template_output(rendered, images, return_tensors, processor_kwargs)
            return self.tokenizer(rendered, return_tensors=return_tensors, **processor_kwargs.get("text_kwargs", {}))
        return rendered

    def _as_image_list(self, images: Any) -> list[Any]:
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def _as_pil_image(self, image: Any) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Any-resolution preprocessing currently expects PIL.Image inputs.")
        return image.convert("RGB")

    def _resize_and_pad(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        width, height = image.size
        scale = min(target_width / width, target_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def _divide_to_patches(self, image: Image.Image, patch_height: int, patch_width: int) -> list[Image.Image]:
        width, height = image.size
        patches = []
        for top in range(0, height, patch_height):
            for left in range(0, width, patch_width):
                patches.append(image.crop((left, top, left + patch_width, top + patch_height)))
        return patches

    def _preprocess_anyres_images(
        self,
        images: Any,
        return_tensors: str | None,
        image_kwargs: dict[str, Any],
    ) -> BatchFeature:
        if self.image_grid_pinpoints is None:
            raise ValueError("image_grid_pinpoints is required when image_mode='anyres'.")
        image_height, image_width = self._processed_image_size()
        grouped_images: list[list[Image.Image]] = []
        image_sizes: list[list[int]] = []
        for raw_image in self._as_image_list(images):
            image = self._as_pil_image(raw_image)
            width, height = image.size
            image_sizes.append([height, width])
            best_height, best_width = select_best_resolution([height, width], self.image_grid_pinpoints)
            padded = self._resize_and_pad(image, best_height, best_width)
            grouped_images.append([image, *self._divide_to_patches(padded, image_height, image_width)])

        flattened_images = [patch for group in grouped_images for patch in group]
        processed = self.image_processor(flattened_images, return_tensors="pt", **image_kwargs)
        flat_pixel_values = processed["pixel_values"]
        max_patches = max(len(group) for group in grouped_images)
        padded_pixel_values = []
        offset = 0
        for group in grouped_images:
            count = len(group)
            image_pixel_values = flat_pixel_values[offset : offset + count]
            offset += count
            if count < max_patches:
                padding = torch.zeros(
                    (max_patches - count, *image_pixel_values.shape[1:]),
                    dtype=image_pixel_values.dtype,
                    device=image_pixel_values.device,
                )
                image_pixel_values = torch.cat([image_pixel_values, padding], dim=0)
            padded_pixel_values.append(image_pixel_values)

        data = {
            "pixel_values": torch.stack(padded_pixel_values, dim=0),
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(
        self,
        images: Any | None = None,
        text: str | list[str] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You must provide images, text, or both.")

        image_inputs = {}
        if images is not None:
            image_kwargs = dict(kwargs.pop("images_kwargs", {}))
            if self.image_mode == "anyres":
                image_inputs = self._preprocess_anyres_images(images, return_tensors, image_kwargs)
            else:
                image_inputs = self.image_processor(images, return_tensors=return_tensors, **image_kwargs)

        text_inputs = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            if self.image_mode == "anyres":
                image_sizes = image_inputs.get("image_sizes") if image_inputs else None
                image_sizes_list = image_sizes.tolist() if image_sizes is not None else None
                image_size_iter = iter(image_sizes_list or [])
                prompt_strings = [self._expand_image_tokens(sample, image_size_iter) for sample in text]
            else:
                prompt_strings = [self._expand_image_tokens(sample) for sample in text]
            nested_text_kwargs = dict(kwargs.pop("text_kwargs", {}))
            text_kwargs = {**kwargs, **nested_text_kwargs}
            text_inputs = self.tokenizer(prompt_strings, return_tensors=return_tensors, **text_kwargs)

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = ["LlavaAnythingProcessor"]
