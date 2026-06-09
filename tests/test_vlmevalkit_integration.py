from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
VLMEVALKIT_ROOT = REPO_ROOT / "eval" / "VLMEvalKit"
PARROT_CONFIG = REPO_ROOT / "configs" / "eval" / "vlmevalkit_llava_anything_parrot.json"
PARROT_SBATCH = REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_llava_anything_vlmevalkit_parrot_1gpu.sbatch"
NATIVE_PARROT_CONFIG = REPO_ROOT / "configs" / "eval" / "vlmevalkit_parrot_7b_parrot.json"
NATIVE_PARROT_SBATCH = REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_parrot_7b_vlmevalkit_parrot_1gpu.sbatch"

if str(VLMEVALKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMEVALKIT_ROOT))


def test_vlmevalkit_adapter_generates_with_project_inference(monkeypatch, tmp_path: Path) -> None:
    from vlmeval.vlm.llava_anything import LlavaAnything

    calls = []

    def fake_generate_responses(**kwargs):
        calls.append(kwargs)
        return ["The answer is C."]

    fake_model = SimpleNamespace(eval=lambda: None, to=lambda device: None, device="cpu")
    monkeypatch.setattr("vlmeval.vlm.llava_anything.generate_responses", fake_generate_responses)
    monkeypatch.setattr("vlmeval.vlm.llava_anything.AutoProcessor.from_pretrained", lambda *args, **kwargs: "processor")
    monkeypatch.setattr(
        "vlmeval.vlm.llava_anything.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    image_path = tmp_path / "question.png"
    Image.new("RGB", (4, 4), "white").save(image_path)

    model = LlavaAnything(
        model_path="fake-checkpoint",
        dtype="none",
        device="cpu",
        device_map="",
        max_new_tokens=7,
        system_prompt="system",
    )

    message = [
        {"type": "text", "value": "Question: Pick one.\nA. zero\nB. one\nC. two"},
        {"type": "image", "value": str(image_path)},
    ]

    assert model.generate_inner(message, dataset="MMMB_en") == "C"
    assert calls[0]["model"] is fake_model
    assert calls[0]["processor"] == "processor"
    assert calls[0]["prompts"] == ["Question: Pick one.\nA. zero\nB. one\nC. two"]
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["max_new_tokens"] == 7
    assert len(calls[0]["images"]) == 1
    assert calls[0]["images"][0].mode == "RGB"


def test_vlmevalkit_adapter_batches_project_inference(monkeypatch, tmp_path: Path) -> None:
    from vlmeval.vlm.llava_anything import LlavaAnything

    calls = []

    def fake_generate_responses(**kwargs):
        calls.append(kwargs)
        return ["The answer is C.", "D"]

    fake_model = SimpleNamespace(eval=lambda: None, to=lambda device: None, device="cpu")
    monkeypatch.setattr("vlmeval.vlm.llava_anything.generate_responses", fake_generate_responses)
    monkeypatch.setattr("vlmeval.vlm.llava_anything.AutoProcessor.from_pretrained", lambda *args, **kwargs: "processor")
    monkeypatch.setattr(
        "vlmeval.vlm.llava_anything.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    image_paths = []
    for idx in range(2):
        image_path = tmp_path / f"question-{idx}.png"
        Image.new("RGB", (4, 4), "white").save(image_path)
        image_paths.append(image_path)

    model = LlavaAnything(
        model_path="fake-checkpoint",
        dtype="none",
        device="cpu",
        device_map="",
        batch_size=16,
        max_new_tokens=9,
        system_prompt="system",
    )

    messages = [
        [
            {"type": "text", "value": "Question: Pick one.\nA. zero\nB. one\nC. two"},
            {"type": "image", "value": str(image_paths[0])},
        ],
        [
            {"type": "text", "value": "Question: Pick again.\nA. red\nB. blue\nD. green"},
            {"type": "image", "value": str(image_paths[1])},
        ],
    ]

    assert model.batch_size == 16
    assert model.generate_batch(messages, dataset="MMMB_en") == ["C", "D"]
    assert calls[0]["model"] is fake_model
    assert calls[0]["processor"] == "processor"
    assert len(calls[0]["images"]) == 2
    assert [image.mode for image in calls[0]["images"]] == ["RGB", "RGB"]
    assert calls[0]["prompts"] == [
        "Question: Pick one.\nA. zero\nB. one\nC. two",
        "Question: Pick again.\nA. red\nB. blue\nD. green",
    ]
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["max_new_tokens"] == 9


def test_vlmevalkit_adapter_builds_multilingual_mcq_prompt(tmp_path: Path) -> None:
    from vlmeval.vlm.llava_anything import LlavaAnything

    model = LlavaAnything.__new__(LlavaAnything)
    model.dump_image_func = lambda line: [str(tmp_path / "image.png")]

    prompt = model.build_prompt(
        {
            "question": "Qual opcao esta correta?",
            "hint": "",
            "A": "primeira",
            "B": "segunda",
            "C": "terceira",
        },
        dataset="MMMB_pt",
    )

    assert prompt[0]["type"] == "text"
    assert "A. primeira" in prompt[0]["value"]
    assert "Responda diretamente com a letra da opcao" in prompt[0]["value"]
    assert prompt[1:] == [{"type": "image", "value": str(tmp_path / "image.png")}]


def test_vlmevalkit_local_inference_uses_model_batch_hook(tmp_path: Path) -> None:
    from vlmeval.inference import infer_data
    from vlmeval.smp import load

    class DummyDataset:
        dataset_name = "DummyBench"

        def __init__(self) -> None:
            self.data = pd.DataFrame(
                {
                    "index": [10, 11, 12, 13, 14],
                    "question": ["q0", "q1", "q2", "q3", "q4"],
                }
            )

        def __len__(self) -> int:
            return len(self.data)

        def build_prompt(self, line):
            return [{"type": "text", "value": line["question"]}]

        def dump_image(self, line):
            return []

    class BatchModel:
        batch_size = 2

        def __init__(self) -> None:
            self.batches = []
            self.generate_calls = 0

        def set_dump_image(self, dump_image):
            self.dump_image = dump_image

        def use_custom_prompt(self, dataset_name):
            return False

        def generate(self, message, dataset=None):
            self.generate_calls += 1
            return "single"

        def generate_batch(self, messages, dataset=None):
            self.batches.append([message[0]["value"] for message in messages])
            return [f"batch-{message[0]['value']}" for message in messages]

    model = BatchModel()
    out_file = tmp_path / "pred.pkl"

    infer_data(
        model=model,
        model_name="dummy",
        work_dir=str(tmp_path),
        dataset=DummyDataset(),
        out_file=str(out_file),
        verbose=False,
    )

    assert model.generate_calls == 0
    assert model.batches == [["q0", "q1"], ["q2", "q3"], ["q4"]]
    assert load(str(out_file)) == {
        10: "batch-q0",
        11: "batch-q1",
        12: "batch-q2",
        13: "batch-q3",
        14: "batch-q4",
    }


def test_vlmevalkit_config_targets_parrot_benchmark_groups() -> None:
    cfg = json.loads(PARROT_CONFIG.read_text())

    model_cfg = cfg["model"]["llava_anything_apretus_8b_clipl"]
    assert model_cfg["class"] == "LlavaAnything"
    assert model_cfg["model_path"].endswith("checkpoints/llava-1.5-apretus-8b-clipl")
    assert model_cfg["batch_size"] == 16

    assert cfg["data"] == {
        "MMMB": {"class": "ConcatDataset", "dataset": "MMMB"},
        "MTL_MMBench_DEV": {"class": "ConcatDataset", "dataset": "MTL_MMBench_DEV"},
    }


def test_vlmevalkit_parrot_launcher_uses_exact_matching_judge() -> None:
    script = PARROT_SBATCH.read_text()

    assert 'JUDGE="${JUDGE:-exact_matching}"' in script
    assert '--judge "${JUDGE}"' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in script


def test_vlmevalkit_parrot_registry_imports_when_llava_anything_is_unavailable() -> None:
    from vlmeval.config import supported_VLM

    assert "Parrot" in supported_VLM


def test_vlmevalkit_parrot_adapter_passes_configured_vision_tower(monkeypatch) -> None:
    from vlmeval.vlm.parrot import Parrot

    captured = {}

    class FakeParrotMetaForCausalLM:
        @staticmethod
        def build(model_name, model_path, **kwargs):
            captured["model_name"] = model_name
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs

            fake_model = SimpleNamespace(
                cuda=lambda: None,
                get_vision_tower=lambda: SimpleNamespace(image_processor="processor"),
            )
            fake_model.cuda = lambda: fake_model
            fake_tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=2)
            return fake_model, fake_tokenizer, "conversation_formatter"

    modules = {
        "parrot": types.ModuleType("parrot"),
        "parrot.model": types.ModuleType("parrot.model"),
        "parrot.model.conversation_formatter": types.ModuleType("parrot.model.conversation_formatter"),
        "parrot.model.parrot_arch": types.ModuleType("parrot.model.parrot_arch"),
        "parrot.utils": types.ModuleType("parrot.utils"),
        "parrot.utils.constants": types.ModuleType("parrot.utils.constants"),
        "parrot.utils.mm_utils": types.ModuleType("parrot.utils.mm_utils"),
    }
    modules["parrot.model.conversation_formatter"].ConversationFormatter = object
    modules["parrot.model.parrot_arch"].ParrotMetaForCausalLM = FakeParrotMetaForCausalLM
    modules["parrot.utils.constants"].BEGIN_LINE = "begin"
    modules["parrot.utils.constants"].DEFAULT_IMAGE_TOKEN = "<image>"
    modules["parrot.utils.constants"].END_LINE = "end"
    modules["parrot.utils.mm_utils"].process_images = lambda *args, **kwargs: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    Parrot(model_path="local-parrot", mm_vision_tower="/models/openai/clip-vit-large-patch14-336")

    assert captured == {
        "model_name": "parrot_qwen2",
        "model_path": "local-parrot",
        "kwargs": {"mm_vision_tower": "/models/openai/clip-vit-large-patch14-336"},
    }


def test_vlmevalkit_native_parrot_config_targets_local_model_and_benchmarks() -> None:
    cfg = json.loads(NATIVE_PARROT_CONFIG.read_text())

    model_cfg = cfg["model"]["parrot_7b_local"]
    assert model_cfg == {
        "class": "Parrot",
        "model_path": str(REPO_ROOT / "models" / "AIDC-AI" / "Parrot-7B"),
        "mm_vision_tower": str(REPO_ROOT / "models" / "openai" / "clip-vit-large-patch14-336"),
    }
    assert cfg["data"] == {
        "MMMB": {"class": "ConcatDataset", "dataset": "MMMB"},
        "MTL_MMBench_DEV": {"class": "ConcatDataset", "dataset": "MTL_MMBench_DEV"},
    }


def test_vlmevalkit_native_parrot_launcher_uses_isolated_env_and_exact_matching() -> None:
    script = NATIVE_PARROT_SBATCH.read_text()

    assert 'VENV_ACTIVATE="${REPO_ROOT}/.venv-parrot-vlmevalkit/bin/activate"' in script
    assert 'CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/eval/vlmevalkit_parrot_7b_parrot.json}"' in script
    assert 'JUDGE="${JUDGE:-exact_matching}"' in script
    assert '--judge "${JUDGE}"' in script
