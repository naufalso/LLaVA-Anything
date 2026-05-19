from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

from llava_anything.processing_llava_anything import LlavaAnythingProcessor


class DummyImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def preprocess(self, images, return_tensors=None, **kwargs):
        return BatchFeature({"pixel_values": images}, tensor_type=return_tensors)


def tokenizer() -> PreTrainedTokenizerFast:
    raw = Tokenizer(WordLevel({"<unk>": 0, "<image>": 1, "hello": 2}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", additional_special_tokens=["<image>"])


def test_processor_expands_single_image_token() -> None:
    processor = LlavaAnythingProcessor(
        image_processor=DummyImageProcessor(),
        tokenizer=tokenizer(),
        image_token="<image>",
        image_seq_length=3,
    )

    expanded = processor._expand_image_tokens("<image>\nhello")

    assert expanded == "<image><image><image>\nhello"


def test_apply_chat_template_normalizes_multimodal_content() -> None:
    processor = LlavaAnythingProcessor(
        image_processor=DummyImageProcessor(),
        tokenizer=tokenizer(),
        image_token="<image>",
        image_seq_length=3,
    )
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hello"}]}]

    rendered = processor.apply_chat_template(conversation, tokenize=False)

    assert "<image>" in rendered
    assert "hello" in rendered
