from __future__ import annotations

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import CLIPImageProcessor, PreTrainedTokenizerFast, pipeline

from llava_anything import LlavaAnythingForConditionalGeneration, LlavaAnythingProcessor
from test_config_and_model import tiny_config


def pipeline_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0, "hello": 1}
    vocab.update({f"token_{idx}": idx for idx in range(2, 63)})
    vocab["<image>"] = 63
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        additional_special_tokens=["<image>"],
    )


def test_image_text_to_text_pipeline_accepts_tiny_model_and_processor() -> None:
    torch.manual_seed(0)
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    processor = LlavaAnythingProcessor(
        image_processor=CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}),
        tokenizer=pipeline_tokenizer(),
        image_token="<image>",
        image_seq_length=4,
        patch_size=4,
    )
    pipe = pipeline("image-text-to-text", model=model, processor=processor)

    output = pipe(
        {"images": Image.new("RGB", (8, 8), color="white"), "text": "<image> hello"},
        max_new_tokens=1,
    )

    assert isinstance(output, list)
    assert "generated_text" in output[0]



def test_image_text_to_text_pipeline_accepts_chat_images() -> None:
    torch.manual_seed(0)
    model = LlavaAnythingForConditionalGeneration(tiny_config())
    processor = LlavaAnythingProcessor(
        image_processor=CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}),
        tokenizer=pipeline_tokenizer(),
        image_token="<image>",
        image_seq_length=4,
        patch_size=4,
    )
    pipe = pipeline("image-text-to-text", model=model, processor=processor)

    output = pipe(
        text=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (8, 8), color="white")},
                    {"type": "text", "text": "hello"},
                ],
            }
        ],
        max_new_tokens=1,
    )

    assert isinstance(output, list)
    assert "generated_text" in output[0]
