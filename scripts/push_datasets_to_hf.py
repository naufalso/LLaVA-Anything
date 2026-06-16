#!/usr/bin/env python3
"""Export LLaVA-style JSON/JSONL datasets as Hugging Face Datasets.

The script keeps the original LLaVA conversation fields, resolves each
``image`` value relative to an image root, casts that column to
``datasets.Image(decode=False)``, and can save the result locally or upload it
with ``Dataset.push_to_hub``.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


DEFAULT_IMAGE_COLUMN = "image"
DEFAULT_IMAGE_PATH_COLUMN = "image_path"
DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class ExportSpec:
    """Description of one LLaVA dataset export target."""

    data_path: Path
    image_folder: Path
    split: str = DEFAULT_SPLIT
    config_name: str = "default"
    max_samples: Optional[int] = None
    available_images_only: Optional[bool] = None
    require_image: Optional[bool] = None


@dataclass(frozen=True)
class ExportOptions:
    """Record conversion options shared by all export specs."""

    image_column: str = DEFAULT_IMAGE_COLUMN
    image_path_column: Optional[str] = DEFAULT_IMAGE_PATH_COLUMN
    add_record_index: bool = True
    available_images_only: bool = True
    strict_images: bool = False
    require_image: bool = False


@dataclass(frozen=True)
class ExportStats:
    """Summary counts for one converted dataset."""

    read: int = 0
    written: int = 0
    skipped_missing_image: int = 0
    skipped_text_only: int = 0


PRESET_SPECS = {
    "pretrain": ExportSpec(
        data_path=Path("data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"),
        image_folder=Path("data/LLaVA-Pretrain"),
        split=DEFAULT_SPLIT,
        config_name="pretrain",
    ),
    "instruct": ExportSpec(
        data_path=Path("data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"),
        image_folder=Path("data/LLaVA-Instruct-150K"),
        split=DEFAULT_SPLIT,
        config_name="instruct",
    ),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def optional_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def iter_records(path: Path, max_samples: Optional[int] = None) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield object records from a JSON list or JSONL file."""

    path = Path(path)
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative.")

    if path.suffix == ".jsonl":
        yielded = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if max_samples is not None and yielded >= max_samples:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object record at {path}:{line_number}.")
                yield yielded, record
                yielded += 1
        return

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    limit = len(records) if max_samples is None else min(max_samples, len(records))
    for index, record in enumerate(records[:limit]):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object record at {path} index {index}.")
        yield index, record


def _resolve_image_path(image_folder: Path, image_value: Any) -> tuple[str, Path]:
    """Return the original image name and resolved local image path."""

    if isinstance(image_value, Path):
        image_name = str(image_value)
    elif isinstance(image_value, str):
        image_name = image_value
    else:
        raise ValueError(
            "LLaVA image values must be strings relative to the image folder; "
            f"got {type(image_value).__name__}."
        )
    return image_name, image_folder / image_name


def prepare_record(
    record: dict[str, Any],
    image_folder: Path,
    record_index: int,
    options: ExportOptions,
) -> Optional[dict[str, Any]]:
    """Convert one LLaVA record into a Hugging Face Datasets-friendly record."""

    converted = dict(record)
    image_value = record.get(options.image_column)

    if not image_value:
        if options.require_image:
            return None
        converted[options.image_column] = None
        if options.image_path_column and options.image_path_column != options.image_column:
            converted[options.image_path_column] = None
        if options.add_record_index:
            converted["record_index"] = record_index
        return converted

    image_name, image_path = _resolve_image_path(Path(image_folder), image_value)
    if not image_path.is_file():
        if options.strict_images:
            raise FileNotFoundError(f"Image for record {record_index} not found: {image_path}")
        if options.available_images_only:
            return None

    converted[options.image_column] = str(image_path.resolve(strict=False))
    if options.image_path_column and options.image_path_column != options.image_column:
        converted[options.image_path_column] = image_name
    if options.add_record_index:
        converted["record_index"] = record_index
    return converted


def collect_records(
    spec: ExportSpec,
    options: ExportOptions,
    max_samples: Optional[int] = None,
) -> tuple[list[dict[str, Any]], ExportStats]:
    """Load and convert all records for one export spec."""

    effective_max_samples = spec.max_samples if max_samples is None else max_samples
    records: list[dict[str, Any]] = []
    read = 0
    skipped_missing_image = 0
    skipped_text_only = 0

    for record_index, record in iter_records(spec.data_path, max_samples=effective_max_samples):
        read += 1
        has_image = bool(record.get(options.image_column))
        converted = prepare_record(
            record,
            image_folder=spec.image_folder,
            record_index=record_index,
            options=options,
        )
        if converted is None:
            if not has_image and options.require_image:
                skipped_text_only += 1
            else:
                skipped_missing_image += 1
            continue
        records.append(converted)

    return records, ExportStats(
        read=read,
        written=len(records),
        skipped_missing_image=skipped_missing_image,
        skipped_text_only=skipped_text_only,
    )


def spec_from_training_yaml(
    yaml_path: Path,
    split: str = DEFAULT_SPLIT,
    config_name: Optional[str] = None,
) -> ExportSpec:
    """Build an export spec from a LLaVA-Anything training YAML data section."""

    with Path(yaml_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {yaml_path} to contain a YAML mapping.")

    data_section = data.get("data")
    if not isinstance(data_section, dict):
        raise ValueError(f"{yaml_path} must contain a mapping at data.")
    if "data_path" not in data_section:
        raise ValueError(f"{yaml_path} is missing data.data_path.")
    if "image_folder" not in data_section:
        raise ValueError(f"{yaml_path} is missing data.image_folder.")

    raw_max_samples = data_section.get("max_samples")
    max_samples = int(raw_max_samples) if raw_max_samples is not None else None
    return ExportSpec(
        data_path=Path(str(data_section["data_path"])),
        image_folder=Path(str(data_section["image_folder"])),
        split=split,
        config_name=config_name or Path(yaml_path).stem,
        max_samples=max_samples,
        available_images_only=bool(data_section.get("available_images_only", True)),
        require_image=bool(data_section.get("require_image", False)),
    )


def _load_datasets_module():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "This exporter requires Hugging Face Datasets. Install it in the "
            "active environment with: pip install 'datasets[vision]'"
        ) from exc
    return datasets


def build_hf_dataset(
    records: list[dict[str, Any]],
    image_column: str = DEFAULT_IMAGE_COLUMN,
    datasets_module: Any = None,
) -> Any:
    """Create a Hugging Face Dataset and cast the image column without decoding."""

    datasets = datasets_module or _load_datasets_module()
    try:
        dataset = datasets.Dataset.from_list(records, on_mixed_types="use_json")
    except TypeError:
        dataset = datasets.Dataset.from_list(records)
    if records and image_column in records[0]:
        dataset = dataset.cast_column(image_column, datasets.Image(decode=False))
    return dataset


def push_dataset(
    dataset: Any,
    repo_id: str,
    config_name: str,
    split: str,
    private: Optional[bool] = None,
    token: Any = None,
    revision: Optional[str] = None,
    create_pr: bool = False,
    max_shard_size: Optional[str] = None,
    num_shards: Optional[int] = None,
    num_proc: Optional[int] = None,
    commit_message: Optional[str] = None,
) -> Any:
    """Upload one Dataset split/config to the Hub."""

    kwargs: dict[str, Any] = {
        "config_name": config_name,
        "split": split,
        "create_pr": create_pr,
        "embed_external_files": True,
    }
    optional_values = {
        "private": private,
        "token": token,
        "revision": revision,
        "max_shard_size": max_shard_size,
        "num_shards": num_shards,
        "num_proc": num_proc,
        "commit_message": commit_message,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    return dataset.push_to_hub(repo_id, **kwargs)


def save_dataset(dataset: Any, output_dir: Path, spec: ExportSpec) -> Path:
    """Save one converted dataset split below an output directory."""

    target = output_dir / spec.config_name / spec.split
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target))
    return target


def _spec_with_overrides(
    spec: ExportSpec,
    split: Optional[str],
    config_name: Optional[str],
    max_samples: Optional[int],
) -> ExportSpec:
    return ExportSpec(
        data_path=spec.data_path,
        image_folder=spec.image_folder,
        split=split or spec.split,
        config_name=config_name or spec.config_name,
        max_samples=spec.max_samples if max_samples is None else max_samples,
        available_images_only=spec.available_images_only,
        require_image=spec.require_image,
    )


def _expand_presets(presets: list[str]) -> list[ExportSpec]:
    specs: list[ExportSpec] = []
    for preset in presets:
        if preset == "all":
            specs.extend(PRESET_SPECS.values())
        else:
            specs.append(PRESET_SPECS[preset])
    return specs


def build_export_specs(args: argparse.Namespace) -> list[ExportSpec]:
    """Build all export specs requested by the CLI."""

    specs: list[ExportSpec] = []
    if args.preset:
        specs.extend(_expand_presets(args.preset))
    if args.training_yaml:
        specs.extend(
            spec_from_training_yaml(path, split=args.split, config_name=args.config_name)
            for path in args.training_yaml
        )
    if args.data_path or args.image_folder:
        if not args.data_path or not args.image_folder:
            raise ValueError("--data-path and --image-folder must be passed together.")
        specs.append(
            ExportSpec(
                data_path=args.data_path,
                image_folder=args.image_folder,
                split=args.split,
                config_name=args.config_name or args.data_path.parent.name or "default",
                max_samples=args.max_samples,
            )
        )

    if not specs:
        raise ValueError("Choose a source with --preset, --training-yaml, or --data-path/--image-folder.")

    if args.config_name and len(specs) > 1 and not args.training_yaml:
        raise ValueError("--config-name can only be used with one source unless using --training-yaml.")

    return [
        _spec_with_overrides(
            spec,
            split=args.split,
            config_name=args.config_name if len(specs) == 1 else None,
            max_samples=args.max_samples,
        )
        for spec in specs
    ]


def options_for_spec(args: argparse.Namespace, spec: ExportSpec) -> ExportOptions:
    """Resolve CLI and training-YAML options for one spec."""

    available_images_only = (
        spec.available_images_only if spec.available_images_only is not None else True
    )
    if args.keep_missing_images:
        available_images_only = False

    require_image = spec.require_image if spec.require_image is not None else False
    if args.require_image is not None:
        require_image = args.require_image

    return ExportOptions(
        image_column=args.image_column,
        image_path_column=args.image_path_column,
        add_record_index=not args.no_record_index,
        available_images_only=available_images_only,
        strict_images=args.strict_images,
        require_image=require_image,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LLaVA-style JSON/JSONL records into Hugging Face Datasets "
            "with an Image column and optionally push them to the Hub."
        )
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=(*PRESET_SPECS.keys(), "all"),
        help="Use a built-in LLaVA data layout. Can be repeated. 'all' exports all presets.",
    )
    parser.add_argument(
        "--training-yaml",
        action="append",
        type=Path,
        help="Read data.data_path and data.image_folder from a training YAML.",
    )
    parser.add_argument("--data-path", type=Path, help="LLaVA JSON/JSONL annotation file.")
    parser.add_argument("--image-folder", type=Path, help="Root directory for relative image paths.")
    parser.add_argument("--repo-id", help="Destination Hub dataset repo, for example 'org/name'.")
    parser.add_argument("--config-name", help="Hub dataset config/subset name.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split name. Default: {DEFAULT_SPLIT}")
    parser.add_argument("--max-samples", type=optional_non_negative_int, help="Limit records per source.")
    parser.add_argument(
        "--image-column",
        default=DEFAULT_IMAGE_COLUMN,
        help=f"Record column containing image paths. Default: {DEFAULT_IMAGE_COLUMN}",
    )
    parser.add_argument(
        "--image-path-column",
        default=DEFAULT_IMAGE_PATH_COLUMN,
        help=f"Column used to preserve the original relative image path. Default: {DEFAULT_IMAGE_PATH_COLUMN}",
    )
    parser.add_argument(
        "--no-image-path-column",
        action="store_true",
        help="Do not preserve the original relative image path in a separate column.",
    )
    parser.add_argument(
        "--no-record-index",
        action="store_true",
        help="Do not add a record_index column with the source-record index.",
    )
    parser.add_argument(
        "--keep-missing-images",
        action="store_true",
        help="Keep records whose image file is missing instead of skipping them.",
    )
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help="Fail on the first missing image instead of skipping it.",
    )
    require_group = parser.add_mutually_exclusive_group()
    require_group.add_argument(
        "--require-image",
        dest="require_image",
        action="store_true",
        help="Drop text-only records.",
    )
    require_group.add_argument(
        "--allow-text-only",
        dest="require_image",
        action="store_false",
        help="Keep text-only records.",
    )
    parser.set_defaults(require_image=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save converted datasets locally below this directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize records without building, saving, or pushing a Dataset.",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument(
        "--token",
        nargs="?",
        const=True,
        default=None,
        help="Hub token or true flag for the cached token.",
    )
    parser.add_argument("--revision", help="Hub branch or revision.")
    parser.add_argument("--create-pr", action="store_true", help="Open a Hub pull request instead of committing directly.")
    parser.add_argument("--max-shard-size", help="Maximum size of each uploaded Parquet shard, such as '1GB'.")
    parser.add_argument("--num-shards", type=positive_int, help="Number of Parquet shards to write.")
    parser.add_argument("--num-proc", type=positive_int, help="Number of processes for upload preparation.")
    parser.add_argument("--commit-message", help="Hub commit message.")
    args = parser.parse_args(argv)
    if args.no_image_path_column:
        args.image_path_column = None
    if args.strict_images and args.keep_missing_images:
        parser.error("--strict-images and --keep-missing-images are mutually exclusive.")
    if not args.dry_run and not args.output_dir and not args.repo_id:
        parser.error("Pass --repo-id, --output-dir, or --dry-run.")
    return args


def run_export(args: argparse.Namespace) -> None:
    specs = build_export_specs(args)
    for spec in specs:
        options = options_for_spec(args, spec)
        print(
            f"Loading {spec.data_path} with images from {spec.image_folder} "
            f"as config={spec.config_name!r} split={spec.split!r}",
            flush=True,
        )
        records, stats = collect_records(spec, options)
        print(
            "Prepared "
            f"{stats.written:,}/{stats.read:,} records "
            f"(skipped_missing_image={stats.skipped_missing_image:,}, "
            f"skipped_text_only={stats.skipped_text_only:,}).",
            flush=True,
        )

        if args.dry_run:
            continue

        dataset = build_hf_dataset(records, image_column=options.image_column)
        if args.output_dir:
            saved_path = save_dataset(dataset, args.output_dir, spec)
            print(f"Saved {spec.config_name}/{spec.split} to {saved_path}", flush=True)
        if args.repo_id:
            url = push_dataset(
                dataset,
                repo_id=args.repo_id,
                config_name=spec.config_name,
                split=spec.split,
                private=args.private or None,
                token=args.token,
                revision=args.revision,
                create_pr=args.create_pr,
                max_shard_size=args.max_shard_size,
                num_shards=args.num_shards,
                num_proc=args.num_proc,
                commit_message=args.commit_message,
            )
            print(f"Pushed {spec.config_name}/{spec.split}: {url}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run_export(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
