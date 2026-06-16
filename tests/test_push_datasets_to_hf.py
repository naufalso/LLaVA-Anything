from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_exporter_module():
    script_path = ROOT / "scripts" / "push_datasets_to_hf.py"
    spec = importlib.util.spec_from_file_location("push_datasets_to_hf", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_record_file(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_iter_records_reads_json_and_jsonl_with_max_samples(tmp_path: Path) -> None:
    exporter = load_exporter_module()
    json_path = tmp_path / "records.json"
    jsonl_path = tmp_path / "records.jsonl"
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    write_record_file(json_path, records)
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    assert [record for _, record in exporter.iter_records(json_path, max_samples=2)] == records[:2]
    assert [record for _, record in exporter.iter_records(jsonl_path, max_samples=2)] == records[:2]


def test_prepare_record_resolves_image_and_preserves_relative_path(tmp_path: Path) -> None:
    exporter = load_exporter_module()
    image_folder = tmp_path / "images"
    image_path = image_folder / "nested" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image bytes")
    options = exporter.ExportOptions()

    converted = exporter.prepare_record(
        {
            "id": "sample-1",
            "image": "nested/sample.jpg",
            "conversations": [{"from": "human", "value": "<image>\nDescribe it."}],
        },
        image_folder=image_folder,
        record_index=7,
        options=options,
    )

    assert converted is not None
    assert converted["image"] == str(image_path.resolve())
    assert converted["image_path"] == "nested/sample.jpg"
    assert converted["record_index"] == 7
    assert converted["id"] == "sample-1"


def test_prepare_record_skips_missing_images_by_default(tmp_path: Path) -> None:
    exporter = load_exporter_module()
    options = exporter.ExportOptions()

    converted = exporter.prepare_record(
        {"id": "missing", "image": "missing.jpg"},
        image_folder=tmp_path,
        record_index=0,
        options=options,
    )

    assert converted is None


def test_prepare_record_can_require_images_for_text_only_records(tmp_path: Path) -> None:
    exporter = load_exporter_module()
    keep_text_options = exporter.ExportOptions(require_image=False)
    drop_text_options = exporter.ExportOptions(require_image=True)

    kept = exporter.prepare_record(
        {"id": "text-only", "conversations": []},
        image_folder=tmp_path,
        record_index=1,
        options=keep_text_options,
    )
    dropped = exporter.prepare_record(
        {"id": "text-only", "conversations": []},
        image_folder=tmp_path,
        record_index=1,
        options=drop_text_options,
    )

    assert kept is not None
    assert kept["image"] is None
    assert kept["image_path"] is None
    assert dropped is None


def test_spec_from_training_yaml_reads_data_section(tmp_path: Path) -> None:
    exporter = load_exporter_module()
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_path": "data/train.json",
                    "image_folder": "data/images",
                    "max_samples": 123,
                    "require_image": True,
                }
            }
        ),
        encoding="utf-8",
    )

    spec = exporter.spec_from_training_yaml(
        config_path,
        split="validation",
        config_name="llava-instruct",
    )

    assert spec.data_path == Path("data/train.json")
    assert spec.image_folder == Path("data/images")
    assert spec.max_samples == 123
    assert spec.require_image is True
    assert spec.split == "validation"
    assert spec.config_name == "llava-instruct"


def test_build_hf_dataset_casts_image_column_without_decoding() -> None:
    exporter = load_exporter_module()
    calls = []

    class FakeImage:
        def __init__(self, decode: bool = True) -> None:
            self.decode = decode

    class FakeDataset:
        @classmethod
        def from_list(cls, records, **kwargs):
            calls.append(("from_list", records, kwargs))
            return cls()

        def cast_column(self, column_name: str, feature) -> "FakeDataset":
            calls.append(("cast_column", column_name, feature.decode))
            return self

    fake_datasets = SimpleNamespace(Dataset=FakeDataset, Image=FakeImage)

    dataset = exporter.build_hf_dataset(
        [{"image": "/tmp/image.jpg", "image_path": "image.jpg"}],
        datasets_module=fake_datasets,
    )

    assert isinstance(dataset, FakeDataset)
    assert calls == [
        (
            "from_list",
            [{"image": "/tmp/image.jpg", "image_path": "image.jpg"}],
            {"on_mixed_types": "use_json"},
        ),
        ("cast_column", "image", False),
    ]


def test_push_dataset_passes_hub_options() -> None:
    exporter = load_exporter_module()
    calls = []

    class FakeDataset:
        def push_to_hub(self, repo_id: str, **kwargs):
            calls.append((repo_id, kwargs))
            return "https://huggingface.co/datasets/user/dataset"

    result = exporter.push_dataset(
        FakeDataset(),
        repo_id="user/dataset",
        config_name="pretrain",
        split="train",
        private=True,
        token=True,
        revision="main",
        create_pr=True,
        max_shard_size="1GB",
        num_shards=4,
        num_proc=2,
        commit_message="Upload LLaVA data",
    )

    assert result == "https://huggingface.co/datasets/user/dataset"
    assert calls == [
        (
            "user/dataset",
            {
                "config_name": "pretrain",
                "split": "train",
                "private": True,
                "token": True,
                "revision": "main",
                "create_pr": True,
                "max_shard_size": "1GB",
                "num_shards": 4,
                "num_proc": 2,
                "commit_message": "Upload LLaVA data",
                "embed_external_files": True,
            },
        )
    ]
