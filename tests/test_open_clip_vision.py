from __future__ import annotations

import builtins
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llava_anything.builder import config_from_yaml_dict, model_from_yaml_dict, processor_from_yaml_dict
from llava_anything.open_clip_vision import OpenCLIPVisionConfig, OpenCLIPVisionTower


class FakeOpenCLIPVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward_intermediates(self, pixel_values, **kwargs):
        batch_size = pixel_values.shape[0]
        patch_tokens = torch.ones(batch_size, 4, 12, device=pixel_values.device, dtype=pixel_values.dtype)
        cls_tokens = torch.zeros(batch_size, 1, 12, device=pixel_values.device, dtype=pixel_values.dtype)
        return {
            "image_intermediates": [patch_tokens, patch_tokens * 2],
            "image_intermediates_prefix": [cls_tokens, cls_tokens + 3],
        }


def test_open_clip_vision_tower_returns_clip_like_hidden_states() -> None:
    config = OpenCLIPVisionConfig(image_size=8, patch_size=4, hidden_size=12, num_hidden_layers=2)
    tower = OpenCLIPVisionTower(config, visual=FakeOpenCLIPVisual())

    outputs = tower(torch.randn(2, 3, 8, 8), output_hidden_states=True, return_dict=True)

    assert len(outputs.hidden_states) == 2
    assert outputs.hidden_states[-1].shape == (2, 5, 12)
    assert torch.all(outputs.hidden_states[-1][:, :1] == 3)
    assert torch.all(outputs.hidden_states[-1][:, 1:] == 2)


def _write_open_clip_repo(path) -> None:
    path.mkdir()
    (path / "open_clip_config.json").write_text(
        json.dumps(
            {
                "model_cfg": {
                    "embed_dim": 32,
                    "vision_cfg": {
                        "image_size": 8,
                        "patch_size": 4,
                        "width": 12,
                        "layers": 2,
                        "head_width": 4,
                        "mlp_ratio": 4.0,
                    },
                    "text_cfg": {
                        "context_length": 77,
                        "vocab_size": 10,
                        "width": 8,
                        "heads": 2,
                        "layers": 1,
                    },
                    "quick_gelu": True,
                },
                "preprocess_cfg": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]},
            }
        ),
        encoding="utf-8",
    )


def _open_clip_yaml(text_dir, vision_dir):
    return {
        "model": {
            "image_token": "<image>",
            "image_token_index": 63,
            "projector_type": "linear",
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "default",
        },
        "text_model": {"name_or_path": str(text_dir), "tokenizer": {}},
        "vision_model": {
            "backend": "open_clip",
            "name_or_path": str(vision_dir),
            "image_processor": {"patch_size": 4},
        },
    }


def test_builder_creates_open_clip_config_and_processor(tmp_path, tiny_text_component_dir) -> None:
    vision_dir = tmp_path / "openclip"
    _write_open_clip_repo(vision_dir)
    data = _open_clip_yaml(tiny_text_component_dir, vision_dir)

    config = config_from_yaml_dict(data)
    processor = processor_from_yaml_dict(data, config)

    assert config.vision_config.model_type == "open_clip_vision_model"
    assert config.vision_config.hidden_size == 12
    assert config.vision_config.image_size == 8
    assert config.vision_config.patch_size == 4
    assert config.image_seq_length == 4
    assert list(processor.image_processor.image_mean) == [0.1, 0.2, 0.3]
    assert list(processor.image_processor.image_std) == [0.4, 0.5, 0.6]


def test_model_from_yaml_loads_open_clip_vision_tower(monkeypatch, tmp_path, tiny_text_component_dir) -> None:
    vision_dir = tmp_path / "openclip"
    _write_open_clip_repo(vision_dir)
    data = _open_clip_yaml(tiny_text_component_dir, vision_dir)
    calls = []

    def fake_create_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return SimpleNamespace(visual=FakeOpenCLIPVisual())

    monkeypatch.setitem(sys.modules, "open_clip", SimpleNamespace(create_model=fake_create_model))

    config = config_from_yaml_dict(data)
    model = model_from_yaml_dict(data, config, load_pretrained_components=True)
    outputs = model(input_ids=torch.tensor([[1, 63, 63, 63, 63, 2]]), pixel_values=torch.randn(1, 3, 8, 8))

    assert calls[0][0] == f"local-dir:{vision_dir}"
    assert isinstance(model.vision_tower, OpenCLIPVisionTower)
    assert outputs.image_hidden_states.shape == (1, 4, 16)


def test_open_clip_missing_dependency_error_points_to_optional_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "open_clip" or name.startswith("open_clip."):
            raise ImportError("missing open_clip")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"llava-anything\[openclip\]"):
        OpenCLIPVisionTower.from_pretrained("hf-hub:example/openclip", OpenCLIPVisionConfig())


def test_default_imports_do_not_load_open_clip_adapter() -> None:
    script = (
        "import sys\n"
        "import llava_anything\n"
        "import llava_anything.modeling_llava_anything\n"
        "import llava_anything.builder\n"
        "raise SystemExit(1 if 'llava_anything.open_clip_vision' in sys.modules else 0)\n"
    )

    subprocess.run([sys.executable, "-c", script], check=True)
