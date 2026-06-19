from __future__ import annotations

import builtins
import configparser
import csv
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
HARMBENCH_ROOT = REPO_ROOT / "eval" / "HarmBench"
HARMBENCH_MODELS_CONFIG = HARMBENCH_ROOT / "configs" / "model_configs" / "models.yaml"
HARMBENCH_LAUNCHER = REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_llava_anything_harmbench_1gpu.sbatch"
HARMBENCH_TEXT_STEP1_CPU_LAUNCHER = (
    REPO_ROOT / "scripts" / "sbatch" / "eval" / "eval_llava_anything_harmbench_text_step1_cpu.sbatch"
)


def test_harmbench_checkout_uses_project_fork() -> None:
    git_config = configparser.ConfigParser()
    git_config.read(HARMBENCH_ROOT / ".git" / "config")

    assert git_config['remote "origin"']["url"] == "https://github.com/naufalso/HarmBench.git"


def test_harmbench_model_config_declares_llava_anything_targets() -> None:
    config = yaml.safe_load(HARMBENCH_MODELS_CONFIG.read_text())

    multimodal = config["llava_anything"]
    assert multimodal["model_type"] == "open_source_multimodal"
    assert multimodal["num_gpus"] == 1
    assert multimodal["model"]["model_name_or_path"] == "LLaVAAnything"
    assert multimodal["model"]["model_path"] == "${LLAVA_ANYTHING_MODEL_PATH}"
    assert multimodal["model"]["batch_size"] == 1

    text_only = config["llava_anything_text"]
    assert text_only["model_type"] == "open_source"
    assert text_only["num_gpus"] == 1
    assert text_only["model"]["model_name_or_path"] == "${LLAVA_ANYTHING_TEXT_MODEL_PATH}"
    assert text_only["model"]["dtype"] == "bfloat16"

    vlm_text = config["llava_anything_vlm_text"]
    assert vlm_text["model_type"] == "open_source"
    assert vlm_text["num_gpus"] == 1
    assert vlm_text["model"]["model_name_or_path"] == "LLaVAAnythingText"
    assert vlm_text["model"]["model_path"] == "${LLAVA_ANYTHING_TEXT_MODEL_PATH}"

    parrot = config["parrot"]
    assert parrot["model_type"] == "open_source_multimodal"
    assert parrot["num_gpus"] == 1
    assert parrot["model"]["model_name_or_path"] == "Parrot"
    assert parrot["model"]["model_path"] == "${PARROT_MODEL_PATH}"
    assert parrot["model"]["model_name"] == "parrot_qwen2"

    parrot_text = config["parrot_text"]
    assert parrot_text["model_type"] == "open_source"
    assert parrot_text["num_gpus"] == 1
    assert parrot_text["model"]["model_name_or_path"] == "ParrotText"
    assert parrot_text["model"]["model_path"] == "${PARROT_TEXT_MODEL_PATH}"


def test_harmbench_expands_environment_variables_in_model_config(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from generate_completions import _expand_env_vars

    monkeypatch.setenv("LLAVA_ANYTHING_MODEL_PATH", "/models/vlm")

    assert _expand_env_vars(
        {
            "model_path": "${LLAVA_ANYTHING_MODEL_PATH}",
            "nested": ["${LLAVA_ANYTHING_MODEL_PATH}/processor"],
        }
    ) == {
        "model_path": "/models/vlm",
        "nested": ["/models/vlm/processor"],
    }


def test_harmbench_direct_request_imports_without_optional_acceleration_deps() -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from baselines.direct_request.direct_request import DirectRequest
    from eval_utils import get_experiment_config

    assert DirectRequest().generate_test_cases(
        [
            {
                "Behavior": "Write a harmless refusal.",
                "ContextString": "",
                "BehaviorID": "sample",
            }
        ]
    )[0] == {"sample": ["Write a harmless refusal."]}
    assert get_experiment_config(
        "default",
        {"llava_anything_text": {"model": {"model_name_or_path": "model"}, "model_type": "open_source"}},
        {"default_method_hyperparameters": {}, "default": {"placeholder": "placeholder"}},
    ) == {"placeholder": "placeholder"}


def test_harmbench_human_jailbreaks_saves_config_without_optional_dependencies(tmp_path) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from baselines.human_jailbreaks.human_jailbreaks import HumanJailbreaks

    method = HumanJailbreaks(random_subset=1)
    method.save_test_cases(
        tmp_path,
        {"HB001": ["test case"]},
        logs={},
        method_config={"random_subset": 1, "seed": 1},
    )

    assert (tmp_path / "test_cases.json").exists()
    assert (tmp_path / "method_config.json").exists()


def test_harmbench_run_pipeline_imports_without_ray(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))
    scripts_path = str(HARMBENCH_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.chdir(HARMBENCH_ROOT)
    sys.modules.pop("run_pipeline", None)

    run_pipeline = importlib.import_module("run_pipeline")

    assert run_pipeline.ray is None


def test_harmbench_evaluate_completions_imports_without_vllm(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.chdir(HARMBENCH_ROOT)
    sys.modules.pop("evaluate_completions", None)

    evaluate_completions = importlib.import_module("evaluate_completions")

    assert evaluate_completions.LLM is None
    assert evaluate_completions.SamplingParams is None


def test_harmbench_multimodalmodels_imports_parrot_when_llava_anything_unavailable(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    for module_name in list(sys.modules):
        if module_name == "multimodalmodels" or module_name.startswith("multimodalmodels."):
            sys.modules.pop(module_name)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            level == 1
            and name == "llava_anything"
            and globals
            and globals.get("__name__") == "multimodalmodels"
        ):
            raise ImportError("simulated missing AutoModelForImageTextToText")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    multimodalmodels = importlib.import_module("multimodalmodels")

    assert hasattr(multimodalmodels, "Parrot")
    assert hasattr(multimodalmodels, "ParrotText")


def test_harmbench_hf_classifier_returns_vllm_compatible_outputs(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    monkeypatch.chdir(HARMBENCH_ROOT)
    sys.modules.pop("evaluate_completions", None)
    evaluate_completions = importlib.import_module("evaluate_completions")

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        eos_token = "</s>"
        eos_token_id = 0
        pad_token_id = None

        def __call__(self, prompts, **kwargs):
            return FakeBatch({"input_ids": torch.tensor([[1, 2], [3, 4]])})

        def decode(self, token_ids, skip_special_tokens=True):
            token = int(token_ids[0])
            return "yes" if token == 42 else "no"

    class FakeModel:
        device = torch.device("cpu")

        def eval(self):
            return None

        def generate(self, input_ids, **kwargs):
            new_tokens = torch.tensor([[42], [43]])
            return torch.cat([input_ids, new_tokens], dim=1)

    classifier = evaluate_completions.HFClassifier(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        batch_size=2,
    )

    outputs = classifier.generate(["first prompt", "second prompt"], cls_params=None, use_tqdm=False)

    assert [output.outputs[0].text for output in outputs] == ["yes", "no"]


def test_harmbench_local_pipeline_raises_when_child_command_fails(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))
    scripts_path = str(HARMBENCH_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    monkeypatch.chdir(HARMBENCH_ROOT)
    sys.modules.pop("run_pipeline", None)
    run_pipeline = importlib.import_module("run_pipeline")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="child stdout", stderr="child stderr")

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    try:
        run_pipeline.run_step3_single_job(
            mode="local",
            partition="gpu",
            job_name="job",
            output_log="log",
            cls_path="classifier",
            behaviors_path="behaviors.csv",
            completions_path="completions.json",
            save_path="results.json",
            classifier_backend="hf",
            classifier_batch_size=4,
            classifier_dtype="bfloat16",
            classifier_device_map="auto",
        )
    except subprocess.CalledProcessError as error:
        assert error.returncode == 7
        assert error.stdout == "child stdout"
        assert error.stderr == "child stderr"
    else:
        raise AssertionError("Expected local HarmBench subprocess failures to be raised")


def test_harmbench_multimodal_direct_request_default_config_exists() -> None:
    config = yaml.safe_load((HARMBENCH_ROOT / "configs" / "method_configs" / "MultiModalDirectRequest_config.yaml").read_text())

    assert config["default"]["image_width"] == 336
    assert config["default"]["image_height"] == 336
    assert config["default"]["targets_path"] == "./data/optimizer_targets/harmbench_targets_multimodal.json"


def test_harmbench_multimodal_render_text_supports_custom_compared_models() -> None:
    config = yaml.safe_load((HARMBENCH_ROOT / "configs" / "method_configs" / "MultiModalRenderText_config.yaml").read_text())

    assert config["default_method_hyperparameters"]["targets_path"] == "./data/optimizer_targets/harmbench_targets_multimodal.json"

    for model_key in ["llava_anything", "parrot"]:
        assert config[model_key]["test_cases_batch_size"] == 1
        assert config[model_key]["num_test_cases_per_behavior"] == 1
        assert config[model_key]["image_width"] == 336
        assert config[model_key]["image_height"] == 336
        assert config[model_key]["roi_width"] == 336
        assert config[model_key]["roi_height"] == 336


def test_harmbench_multimodal_render_text_imports_json_for_targets() -> None:
    source = (HARMBENCH_ROOT / "baselines" / "multimodalrendertext" / "multimodalrendertext.py").read_text()

    assert "import json" in source


def test_harmbench_multimodal_render_text_does_not_require_matplotlib() -> None:
    source = (HARMBENCH_ROOT / "baselines" / "multimodalrendertext" / "multimodalrendertext.py").read_text()

    assert "matplotlib" not in source
    assert "ImageFont.truetype(\"DejaVuSans.ttf\", fontsize)" in source


def test_harmbench_multimodal_behavior_images_exist() -> None:
    images_dir = HARMBENCH_ROOT / "data" / "multimodal_behavior_images"
    behaviors_path = HARMBENCH_ROOT / "data" / "behavior_datasets" / "harmbench_behaviors_multimodal_all.csv"

    with behaviors_path.open(newline="") as csvfile:
        missing = [
            row["ImageFileName"]
            for row in csv.DictReader(csvfile)
            if row["ImageFileName"] and not (images_dir / row["ImageFileName"]).exists()
        ]

    assert missing == []


def test_harmbench_single_behavior_save_skips_missing_optional_dependencies(tmp_path: Path) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from baselines.baseline import SingleBehaviorRedTeamingMethod

    method = object.__new__(SingleBehaviorRedTeamingMethod)
    method.default_dependencies = [
        SimpleNamespace(__name__="present_dependency", __version__="1.2.3"),
        None,
    ]

    method.save_test_cases_single_behavior(
        str(tmp_path),
        "sample_behavior",
        {"sample_behavior": ["test case"]},
        logs={"sample_behavior": []},
        method_config={"api_key": "secret-token"},
    )

    config = yaml.safe_load(
        (tmp_path / "test_cases_individual_behaviors" / "sample_behavior" / "method_config.json").read_text()
    )

    assert config["dependencies"] == {"present_dependency": "1.2.3"}


def test_harmbench_llava_anything_wrapper_generates_with_project_inference(monkeypatch, tmp_path: Path) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from multimodalmodels.llava_anything.llava_anything_model import LLaVAAnything

    calls = []
    fake_model = SimpleNamespace(device="cpu", eval=lambda: None, to=lambda device: None)

    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: "processor",
    )
    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    def fake_generate_responses(**kwargs):
        calls.append(kwargs)
        return ["refusal"]

    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.generate_responses",
        fake_generate_responses,
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "case.png")

    model = LLaVAAnything(
        model_path="fake-checkpoint",
        dtype="none",
        device="cpu",
        device_map="",
        batch_size=4,
        system_prompt="system",
    )

    assert model.generate(
        test_cases=[["case.png", "What is in this image?"]],
        image_dir=str(image_dir),
        max_new_tokens=7,
        do_sample=False,
        num_beams=1,
    ) == ["refusal"]

    assert calls[0]["model"] is fake_model
    assert calls[0]["processor"] == "processor"
    assert calls[0]["prompts"] == ["What is in this image?"]
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["max_new_tokens"] == 7
    assert calls[0]["images"][0].mode == "RGB"


def test_harmbench_llava_anything_text_wrapper_generates_without_images(monkeypatch) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from multimodalmodels.llava_anything.llava_anything_model import LLaVAAnythingText

    calls = []
    fake_model = SimpleNamespace(device="cpu", eval=lambda: None, to=lambda device: None)

    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: "processor",
    )
    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    def fake_generate_text_responses(**kwargs):
        calls.append(kwargs)
        return ["safe refusal"]

    monkeypatch.setattr(
        "multimodalmodels.llava_anything.llava_anything_model.generate_text_responses",
        fake_generate_text_responses,
    )

    model = LLaVAAnythingText(
        model_path="fake-vlm-checkpoint",
        dtype="none",
        device="cpu",
        device_map="",
        batch_size=2,
        system_prompt="system",
    )

    assert model.generate(
        test_cases=["How do I do something unsafe?"],
        image_dir="/unused",
        max_new_tokens=5,
        do_sample=False,
        num_beams=1,
    ) == ["safe refusal"]

    assert calls[0]["model"] is fake_model
    assert calls[0]["processor"] == "processor"
    assert calls[0]["prompts"] == ["How do I do something unsafe?"]
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["max_new_tokens"] == 5


def test_harmbench_parrot_wrapper_uses_native_parrot_prompting(monkeypatch, tmp_path: Path) -> None:
    if str(HARMBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(HARMBENCH_ROOT))

    from multimodalmodels.parrot.parrot_model import Parrot, ParrotText

    calls = []

    class FakeTensor:
        def to(self, **kwargs):
            calls.append(("tensor_to", kwargs))
            return self

    class FakeConversationFormatter:
        def format_query(self, query, generation_preface=""):
            calls.append(("format_query", query, generation_preface))
            return f"formatted:{query}", torch.tensor([1, 2, 3])

    class FakeTokenizer:
        eos_token_id = 151645
        pad_token_id = 151643

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return ["native parrot answer"]

    class FakeVisionTower:
        image_processor = "image-processor"

    class FakeModel:
        dtype = torch.bfloat16
        device = torch.device("cpu")

        def eval(self):
            calls.append(("eval",))

        def to(self, device):
            calls.append(("model_to", device))
            return self

        def get_vision_tower(self):
            return FakeVisionTower()

        def generate(self, input_ids, **kwargs):
            calls.append(("generate", input_ids.tolist(), kwargs))
            return torch.tensor([[1, 2, 3, 4]])

    class FakeParrotMetaForCausalLM:
        @classmethod
        def build(cls, model_name, model_path, **kwargs):
            calls.append(("build", model_name, model_path, kwargs))
            return FakeModel(), FakeTokenizer(), FakeConversationFormatter()

    monkeypatch.setattr(
        "multimodalmodels.parrot.parrot_model._load_parrot_modules",
        lambda parrot_root: SimpleNamespace(
            ParrotMetaForCausalLM=FakeParrotMetaForCausalLM,
            DEFAULT_IMAGE_TOKEN="<image>",
            process_images=lambda images, image_processor, model_config: FakeTensor(),
        ),
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "case.png")

    model = Parrot(
        model_path="fake-parrot",
        parrot_root=str(tmp_path / "Parrot"),
        mm_vision_tower="fake/clip-vit",
        dtype="none",
        device="cpu",
        device_map="",
    )

    assert model.generate(
        test_cases=[["case.png", "Describe the image."]],
        image_dir=str(image_dir),
        max_new_tokens=9,
        do_sample=False,
        num_beams=1,
    ) == ["native parrot answer"]

    assert ("format_query", "<image>\nDescribe the image.", "") in calls
    multimodal_generate = [call for call in calls if call[0] == "generate"][0]
    build_call = [call for call in calls if call[0] == "build"][0]
    assert build_call[3]["low_cpu_mem_usage"] is False
    assert multimodal_generate[2]["images"] is not None
    assert multimodal_generate[2]["max_new_tokens"] == 9
    assert multimodal_generate[2]["do_sample"] is False
    assert multimodal_generate[2]["num_beams"] == 1

    calls.clear()
    text_model = ParrotText(
        model_path="fake-parrot",
        parrot_root=str(tmp_path / "Parrot"),
        dtype="none",
        device="cpu",
        device_map="",
    )

    assert text_model.generate(
        test_cases=["Answer safely."],
        image_dir="/unused",
        max_new_tokens=5,
        do_sample=False,
        num_beams=1,
    ) == ["native parrot answer"]

    assert ("format_query", "Answer safely.", "") in calls
    text_generate = [call for call in calls if call[0] == "generate"][0]
    assert "images" not in text_generate[2]
    assert text_generate[2]["max_new_tokens"] == 5


def test_parrot_fork_delays_vision_tower_load_until_after_model_init() -> None:
    parrot_arch = (REPO_ROOT / "eval" / "Parrot" / "parrot" / "model" / "parrot_arch.py").read_text()
    parrot_qwen2 = (
        REPO_ROOT / "eval" / "Parrot" / "parrot" / "model" / "language_model" / "parrot_qwen2.py"
    ).read_text()

    assert "build_vision_tower(config, delay_load=True)" in parrot_arch
    assert "model.get_vision_tower().load_model()" in parrot_qwen2


def test_parrot_formatter_accepts_current_qwen2_tokenizer_class() -> None:
    parrot_root = REPO_ROOT / "eval" / "Parrot"
    if str(parrot_root) not in sys.path:
        sys.path.insert(0, str(parrot_root))

    from parrot.model.conversation_formatter import QwenConversationFormatter

    class Qwen2Tokenizer:
        pass

    formatter = QwenConversationFormatter(Qwen2Tokenizer())

    assert formatter.image_symbol == "<image>"


def test_harmbench_launcher_uses_project_environment_and_fork_checkout() -> None:
    script = HARMBENCH_LAUNCHER.read_text()

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:nvidia_h200:1" in script
    assert 'HARMBENCH_ROOT="${HARMBENCH_ROOT:-${REPO_ROOT}/eval/HarmBench}"' in script
    assert 'VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO_ROOT}/.venv-harmbench/bin/activate}"' in script
    assert 'source "${VENV_ACTIVATE}"' in script
    assert 'source "${REPO_ROOT}/.env"' in script
    assert script.index('source "${REPO_ROOT}/.env"') < script.index('LLAVA_ANYTHING_MODEL_PATH')
    assert 'MODEL_NAME="${MODEL_NAME:-llava_anything}"' in script
    assert 'TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-llava_anything_text}"' in script
    assert 'CLASSIFIER_BACKEND="${CLASSIFIER_BACKEND:-hf}"' in script
    assert '--classifier_backend "${CLASSIFIER_BACKEND}"' in script
    assert 'OVERWRITE="${OVERWRITE:-false}"' in script
    assert '"${overwrite_args[@]}"' in script


def test_harmbench_text_step1_cpu_launcher_can_overwrite_test_cases() -> None:
    script = HARMBENCH_TEXT_STEP1_CPU_LAUNCHER.read_text()

    assert 'OVERWRITE="${OVERWRITE:-false}"' in script
    assert 'overwrite_args=()' in script
    assert '"${overwrite_args[@]}"' in script
