from __future__ import annotations

import tomllib
from pathlib import Path


def test_accelerate_is_runtime_dependency_for_device_map_auto() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.split(">=", 1)[0] == "accelerate" for dependency in dependencies)


def test_python_support_is_bounded_to_tested_versions() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"


def test_torch_dependency_supports_safe_bin_weight_loading() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert "torch>=2.6,<2.7" in dependencies
    assert "torchvision>=0.21,<0.22" in dependencies
