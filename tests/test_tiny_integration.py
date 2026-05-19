from __future__ import annotations

from transformers import AutoModelForImageTextToText, AutoProcessor

from llava_anything.builder import save_from_yaml


def test_tiny_local_components_build_reload_and_generate(
    tiny_model_yaml_path,
    tiny_full_model_dir,
    tiny_image,
) -> None:
    save_from_yaml(tiny_model_yaml_path, tiny_full_model_dir, load_pretrained_components=True)

    processor = AutoProcessor.from_pretrained(tiny_full_model_dir)
    model = AutoModelForImageTextToText.from_pretrained(tiny_full_model_dir)
    inputs = processor(images=tiny_image, text="<image> hello", return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    assert output.shape[0] == 1
    assert output.shape[1] == inputs["input_ids"].shape[1] + 1
