from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_EVAL_ROOT = REPO_ROOT / "eval" / "lmms-eval"

if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))


def test_llava_anything_eval_adapter_batches_project_inference(monkeypatch) -> None:
    from lmms_eval.api.instance import Instance
    from llava_anything.eval.lmms import LlavaAnything

    calls = []

    def fake_generate_responses(**kwargs):
        calls.append(kwargs)
        return ["Yes" if "white" in prompt else "No" for prompt in kwargs["prompts"]]

    monkeypatch.setattr("llava_anything.eval.lmms.generate_responses", fake_generate_responses)
    monkeypatch.setattr("llava_anything.eval.lmms.AutoProcessor.from_pretrained", lambda *args, **kwargs: "processor")
    monkeypatch.setattr(
        "llava_anything.eval.lmms.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(eval=lambda: None, device="cpu"),
    )

    image = Image.new("RGB", (4, 4), "white")
    other_image = Image.new("RGB", (4, 4), "black")
    model = LlavaAnything(pretrained="fake-checkpoint", dtype="float16", batch_size=2, device_map="", max_new_tokens=8)
    model.task_dict = {"mme": {"test": [{"image": image}, {"image": other_image}]}}

    requests = [
        Instance(
            request_type="generate_until",
            doc={
                "question": "Is this white?",
                "answer": "yes",
                "image": image,
                "category": "color",
                "question_id": "1",
            },
            arguments=(
                "Is this white?\nAnswer the question using a single word or phrase.",
                {"max_new_tokens": 16, "temperature": 0},
                lambda doc: [doc["image"]],
                0,
                "mme",
                "test",
            ),
            idx=0,
            metadata={"task": "mme", "doc_id": 0, "repeats": 1},
        ),
        Instance(
            request_type="generate_until",
            doc={
                "question": "Is this black?",
                "answer": "no",
                "image": other_image,
                "category": "color",
                "question_id": "2",
            },
            arguments=(
                "Is this black?\nAnswer the question using a single word or phrase.",
                {"max_new_tokens": 16, "temperature": 0},
                lambda doc: [doc["image"]],
                1,
                "mme",
                "test",
            ),
            idx=1,
            metadata={"task": "mme", "doc_id": 1, "repeats": 1},
        ),
    ]

    assert model.generate_until(requests) == ["Yes", "No"]
    assert calls[0]["processor"] == "processor"
    assert len(calls[0]["images"]) == 2
    assert image in calls[0]["images"]
    assert other_image in calls[0]["images"]
    assert len(calls[0]["prompts"]) == 2
    assert any(prompt.startswith("Is this white?") for prompt in calls[0]["prompts"])
    assert any(prompt.startswith("Is this black?") for prompt in calls[0]["prompts"])
    assert calls[0]["max_new_tokens"] == 16


def test_llava_anything_eval_adapter_loglikelihood_is_out_of_scope() -> None:
    from llava_anything.eval.lmms import LlavaAnything

    model = LlavaAnything.__new__(LlavaAnything)
    with pytest.raises(NotImplementedError, match="generate_until"):
        model.loglikelihood([])


def test_llava_anything_eval_adapter_sets_left_padding_for_batches() -> None:
    from llava_anything.eval.lmms import _prepare_processor_for_batched_generation

    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token=None,
        pad_token_id=None,
        eos_token="<eos>",
        eos_token_id=2,
    )

    _prepare_processor_for_batched_generation(SimpleNamespace(tokenizer=tokenizer))

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.pad_token_id == 2


def test_llava_anything_eval_adapter_declares_model_entrypoint_manifest() -> None:
    from lmms_eval.models.registry_v2 import ModelManifest
    from llava_anything.eval.lmms import manifest

    model_manifest = manifest()

    assert isinstance(model_manifest, ModelManifest)
    assert model_manifest.model_id == "llava_anything"
    assert model_manifest.simple_class_path == "llava_anything.eval.lmms.LlavaAnything"


def test_project_eval_extra_installs_lmms_eval_and_registers_model() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    eval_deps = pyproject["project"]["optional-dependencies"]["eval"]
    assert any(
        dep.startswith("lmms_eval @ git+https://github.com/naufalso/lmms-eval.git")
        for dep in eval_deps
    )
    assert (
        pyproject["project"]["entry-points"]["lmms_eval.models"]["llava_anything"]
        == "llava_anything.eval.lmms:manifest"
    )


def test_lmms_eval_builtin_registry_includes_parrot() -> None:
    from lmms_eval.models import get_model, get_model_manifest

    model_manifest = get_model_manifest("parrot")

    assert model_manifest.simple_class_path == "lmms_eval.models.simple.parrot.Parrot"
    assert get_model("parrot", force_simple=True).__name__ == "Parrot"


def test_parrot_lmms_adapter_builds_model_and_batches_generation(monkeypatch) -> None:
    from lmms_eval.api.instance import Instance

    build_call = {}

    class FakeModel:
        dtype = torch.float32

        def __init__(self) -> None:
            self.config = SimpleNamespace(tokenizer_padding_side="right")
            self.generate_calls = []

        def to(self, device):
            self.device = str(device)
            return self

        def cuda(self):
            self.device = "cuda"
            return self

        def eval(self):
            self.eval_called = True
            return self

        def get_vision_tower(self):
            return SimpleNamespace(image_processor="fake-image-processor")

        def generate(self, input_ids, **kwargs):
            self.generate_calls.append({"input_ids": input_ids, **kwargs})
            return torch.cat(
                [input_ids, torch.full((input_ids.shape[0], 1), 42, dtype=input_ids.dtype)],
                dim=1,
            )

    class FakeTokenizer:
        eos_token_id = 151643
        pad_token_id = 151643

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["A", "B"][: len(ids)]

    class FakeFormatter:
        def __init__(self) -> None:
            self.queries = []

        def format_query(self, query):
            self.queries.append(query)
            return "formatted prompt", torch.tensor([1, 2])

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    fake_formatter = FakeFormatter()

    class FakeParrotMetaForCausalLM:
        @classmethod
        def build(cls, model_name, model_path, **kwargs):
            build_call.update({"model_name": model_name, "model_path": model_path, **kwargs})
            return fake_model, fake_tokenizer, fake_formatter

    def fake_process_images(images, image_processor, model_cfg):
        assert image_processor == "fake-image-processor"
        assert getattr(model_cfg, "image_aspect_ratio") == "pad"
        assert len(images) == 2
        return torch.zeros((2, 3, 4, 4))

    parrot_arch = types.ModuleType("parrot.model.parrot_arch")
    parrot_arch.ParrotMetaForCausalLM = FakeParrotMetaForCausalLM
    constants = types.ModuleType("parrot.utils.constants")
    constants.DEFAULT_IMAGE_TOKEN = "<image>"
    mm_utils = types.ModuleType("parrot.utils.mm_utils")
    mm_utils.process_images = fake_process_images
    monkeypatch.setitem(sys.modules, "parrot.model.parrot_arch", parrot_arch)
    monkeypatch.setitem(sys.modules, "parrot.utils.constants", constants)
    monkeypatch.setitem(sys.modules, "parrot.utils.mm_utils", mm_utils)

    from lmms_eval.models.simple.parrot import Parrot

    image = Image.new("RGB", (4, 4), "white")
    other_image = Image.new("RGB", (4, 4), "black")
    model = Parrot(
        pretrained="/models/parrot",
        mm_vision_tower="/models/clip",
        device="cpu",
        batch_size=2,
        max_new_tokens=8,
    )
    model.task_dict = {"mmbench": {"validation": [{"image": image}, {"image": other_image}]}}

    requests = [
        Instance(
            request_type="generate_until",
            doc={"image": image},
            arguments=(
                "What option is correct?",
                {"max_new_tokens": 5, "temperature": 0, "until": ["\n"]},
                lambda doc: [doc["image"]],
                0,
                "mmbench",
                "validation",
            ),
            idx=0,
            metadata={"task": "mmbench", "doc_id": 0, "repeats": 1},
        ),
        Instance(
            request_type="generate_until",
            doc={"image": other_image},
            arguments=(
                "What option is second?",
                {"max_new_tokens": 5, "temperature": 0, "until": ["\n"]},
                lambda doc: [doc["image"]],
                1,
                "mmbench",
                "validation",
            ),
            idx=1,
            metadata={"task": "mmbench", "doc_id": 1, "repeats": 1},
        ),
    ]

    assert model.generate_until(requests) == ["A", "B"]
    assert build_call == {
        "model_name": "parrot_qwen2",
        "model_path": "/models/parrot",
        "mm_vision_tower": "/models/clip",
        "low_cpu_mem_usage": True,
    }
    assert fake_formatter.queries == [
        "<image>\nWhat option is correct?",
        "<image>\nWhat option is second?",
    ]
    assert len(fake_model.generate_calls) == 1
    assert fake_model.config.tokenizer_padding_side == "left"
    assert fake_model.generate_calls[0]["input_ids"].shape == (2, 2)
    assert fake_model.generate_calls[0]["attention_mask"].shape == (2, 2)
    assert fake_model.generate_calls[0]["max_new_tokens"] == 5
    assert fake_model.generate_calls[0]["do_sample"] is False
    assert fake_model.generate_calls[0]["num_beams"] == 1
    assert fake_model.generate_calls[0]["images"].shape == (2, 3, 4, 4)


def test_parrot_lmms_sbatch_uses_parrot_env_and_apertus_benchmarks() -> None:
    aperture_script = REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_llava_anything_lmms_apertus_array_1gpu.sbatch"
    parrot_script = REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_parrot_7b_lmms_1gpu.sbatch"

    aperture_benchmarks = _default_benchmarks(aperture_script.read_text())
    parrot_text = parrot_script.read_text()
    parrot_benchmarks = _default_benchmarks(parrot_text)

    assert 'VENV_ACTIVATE="${REPO_ROOT}/.venv-parrot-vlmevalkit/bin/activate"' in parrot_text
    assert 'MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/AIDC-AI/Parrot-7B}"' in parrot_text
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in parrot_text
    assert "--model parrot" in parrot_text
    assert [label for label, _task, _limit in parrot_benchmarks] == [
        label for label, _task, _limit in aperture_benchmarks
    ]

    aperture_task_by_label = {label: task for label, task, _limit in aperture_benchmarks}
    parrot_task_by_label = {label: task for label, task, _limit in parrot_benchmarks}
    for label, task in aperture_task_by_label.items():
        if label == "mmbench":
            assert parrot_task_by_label[label] == "mmbench_en_dev_static"
        else:
            assert parrot_task_by_label[label] == task


def _default_benchmarks(script_text: str) -> list[tuple[str, str, str]]:
    line = next(line for line in script_text.splitlines() if line.startswith("BENCHMARKS="))
    value = line.split(":-", 1)[1].removesuffix('}"')
    return [
        tuple(benchmark.split("|"))  # type: ignore[misc]
        for benchmark in value.split(";")
        if benchmark
    ]
