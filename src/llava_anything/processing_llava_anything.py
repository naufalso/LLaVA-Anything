"""Processor for LLaVa-Anything."""

from __future__ import annotations

from typing import Any

from transformers.feature_extraction_utils import BatchFeature
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
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_token = image_token
        self.image_seq_length = image_seq_length
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_additional_image_tokens = num_additional_image_tokens
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        names = list(getattr(self.tokenizer, "model_input_names", ["input_ids", "attention_mask"]))
        if "pixel_values" not in names:
            names.append("pixel_values")
        if "image_sizes" not in names:
            names.append("image_sizes")
        return names

    def _num_image_tokens(self) -> int:
        if self.image_seq_length is not None:
            return self.image_seq_length

        size = getattr(self.image_processor, "crop_size", None) or getattr(self.image_processor, "size", None)
        if self.patch_size is None or size is None:
            raise ValueError("image_seq_length is required when patch_size or image processor size is unknown.")
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge")
            width = size.get("width") or size.get("shortest_edge")
        else:
            height, width = size
        tokens = (int(height) // self.patch_size) * (int(width) // self.patch_size)
        if self.vision_feature_select_strategy == "full":
            tokens += self.num_additional_image_tokens
        return tokens

    def _expand_image_tokens(self, text: str) -> str:
        count = self._num_image_tokens()
        placeholder = "<llava_anything_image_placeholder>"
        while self.image_token in text:
            text = text.replace(self.image_token, placeholder * count, 1)
        return text.replace(placeholder, self.image_token)

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                parts.append(self.image_token)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    def _normalize_conversation(self, conversation: Any) -> Any:
        if not isinstance(conversation, list):
            return conversation
        normalized = []
        for message in conversation:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            copied = dict(message)
            copied["content"] = self._normalize_content(copied.get("content", ""))
            normalized.append(copied)
        return normalized

    def apply_chat_template(self, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        conversation = self._normalize_conversation(conversation)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
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
        if kwargs.get("tokenize"):
            return self.tokenizer(rendered, return_tensors=kwargs.get("return_tensors"))
        return rendered

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
            image_inputs = self.image_processor(images, return_tensors=return_tensors, **image_kwargs)

        text_inputs = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            prompt_strings = [self._expand_image_tokens(sample) for sample in text]
            text_kwargs = dict(kwargs.pop("text_kwargs", {}))
            text_inputs = self.tokenizer(prompt_strings, return_tensors=return_tensors, **text_kwargs)

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = ["LlavaAnythingProcessor"]
