from __future__ import annotations

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, CLIPImageProcessor, PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

from llava_anything.processing_llava_anything import LlavaAnythingProcessor
from test_config_and_model import tiny_config
from llava_anything import builder


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


def test_anyres_processor_expands_square_wide_and_tall_images() -> None:
    tok = tokenizer()
    tok.pad_token = "<unk>"
    processor = LlavaAnythingProcessor(
        image_processor=CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}),
        tokenizer=tok,
        image_token="<image>",
        patch_size=4,
        image_mode="anyres",
        image_grid_pinpoints=[[8, 8], [8, 16], [16, 8]],
    )
    images = [
        Image.new("RGB", (8, 8)),
        Image.new("RGB", (16, 8)),
        Image.new("RGB", (8, 16)),
    ]

    batch = processor(images=images, text=["<image>", "<image>", "<image>"], return_tensors="pt", padding=True)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert batch["image_sizes"].tolist() == [[8, 8], [8, 16], [16, 8]]
    assert batch["pixel_values"].shape == (3, 3, 3, 8, 8)
    assert (batch["input_ids"] == image_token_id).sum(dim=1).tolist() == [10, 14, 16]


def test_anyres_processor_handles_multiple_image_sizes_in_one_prompt() -> None:
    processor = LlavaAnythingProcessor(
        image_processor=CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}),
        tokenizer=tokenizer(),
        image_token="<image>",
        patch_size=4,
        image_mode="anyres",
        image_grid_pinpoints=[[8, 8], [8, 16]],
    )
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))]

    batch = processor(images=images, text="<image> hello <image>", return_tensors="pt")

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert batch["image_sizes"].tolist() == [[8, 8], [8, 16]]
    assert int((batch["input_ids"] == image_token_id).sum().item()) == 24


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


def test_processor_builder_does_not_shrink_pretrained_vocab(monkeypatch) -> None:
    config = tiny_config()
    config.text_config.vocab_size = 128
    config.vocab_size = 128
    fake_tokenizer = tokenizer()

    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(builder.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: DummyImageProcessor())

    data = {
        "text_model": {"name_or_path": "text", "tokenizer": {}},
        "vision_model": {"name_or_path": "vision", "image_processor": {}},
    }

    builder.processor_from_yaml_dict(data, config)

    assert config.text_config.vocab_size == 128
    assert config.vocab_size == 128


def test_processor_builder_does_not_grow_pretrained_vocab_before_weight_load(monkeypatch) -> None:
    config = tiny_config()
    config.text_config.vocab_size = 3
    config.vocab_size = 3
    raw = Tokenizer(WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    fake_tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")

    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(builder.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: DummyImageProcessor())

    data = {
        "text_model": {"name_or_path": "text", "tokenizer": {}},
        "vision_model": {"name_or_path": "vision", "image_processor": {}},
    }

    builder.processor_from_yaml_dict(data, config)

    assert config.image_token_index == 3
    assert config.text_config.vocab_size == 3
    assert config.vocab_size == 3


def test_auto_processor_reloads_saved_processor(tmp_path) -> None:
    processor = LlavaAnythingProcessor(
        image_processor=CLIPImageProcessor(size={"height": 8, "width": 8}, crop_size={"height": 8, "width": 8}),
        tokenizer=tokenizer(),
        image_token="<image>",
        image_seq_length=4,
        patch_size=4,
    )
    processor.save_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)

    reloaded = AutoProcessor.from_pretrained(tmp_path)

    assert isinstance(reloaded, LlavaAnythingProcessor)
    assert reloaded.image_token == "<image>"
    assert reloaded.image_seq_length == 4


def test_processor_post_process_image_text_to_text_decodes_tokens() -> None:
    processor = LlavaAnythingProcessor(
        image_processor=DummyImageProcessor(),
        tokenizer=tokenizer(),
        image_token="<image>",
        image_seq_length=3,
    )
    encoded = processor.tokenizer(["hello"], return_tensors="pt")

    assert processor.post_process_image_text_to_text(encoded["input_ids"]) == ["hello"]


def test_processor_forwards_direct_text_kwargs_to_tokenizer() -> None:
    tok = tokenizer()
    tok.pad_token = "<unk>"
    processor = LlavaAnythingProcessor(
        image_processor=DummyImageProcessor(),
        tokenizer=tok,
        image_token="<image>",
        image_seq_length=3,
    )

    encoded = processor(text=["hello", "hello hello"], return_tensors="pt", padding=True)

    assert encoded["input_ids"].shape[0] == 2
    assert encoded["input_ids"].shape[1] == 2


def test_processor_builder_sets_tokenizer_model_max_length(monkeypatch) -> None:
    config = tiny_config()
    fake_tokenizer = tokenizer()

    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(builder.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: DummyImageProcessor())

    data = {
        "text_model": {"name_or_path": "text", "tokenizer": {"model_max_length": 1024}},
        "vision_model": {"name_or_path": "vision", "image_processor": {}},
    }

    processor = builder.processor_from_yaml_dict(data, config)

    assert processor.tokenizer.model_max_length == 1024
