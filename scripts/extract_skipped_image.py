#!/usr/bin/env python3
"""Extract records whose images were skipped by the training availability cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CACHE_PATH = Path(
    "checkpoints/apertus-8b-siglip2-anyres-pangea-full/skipped_image_indices.json"
)
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_OUTPUT_NAME = "skipped_image_records.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read skipped_image_indices.json, extract the corresponding dataset "
            "records, and write them to a JSON file for manual missing-image checks."
        )
    )
    parser.add_argument(
        "--skipped-indices",
        default=DEFAULT_CACHE_PATH,
        type=Path,
        help=f"Path to skipped_image_indices.json. Default: {DEFAULT_CACHE_PATH}",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Dataset JSON/JSONL path. Defaults to metadata.data_path from the cache.",
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        help="Image root folder. Defaults to metadata.image_folder from the cache.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for the output JSON file. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help=(
            "Full output JSON path. Overrides --output-dir and defaults to "
            f"--output-dir/{DEFAULT_OUTPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="Write only original records instead of index/path metadata wrappers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Extract at most this many skipped records, useful for quick spot checks.",
    )
    parser.add_argument(
        "--no-verify-cache",
        action="store_true",
        help="Skip stat-based checks against cache metadata.",
    )
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    if not isinstance(cache, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    skipped_indices = cache.get("skipped_indices")
    if not isinstance(skipped_indices, list) or not all(
        isinstance(index, int) for index in skipped_indices
    ):
        raise ValueError(f"{path} does not contain a valid skipped_indices list.")
    return cache


def path_from_cache_or_arg(
    explicit_path: Path | None,
    metadata: dict[str, Any],
    key: str,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    value = metadata.get(key)
    if not value:
        raise ValueError(f"Cache metadata is missing {key}; pass --{key.replace('_', '-')}.")
    return Path(str(value))


def warn_if_cache_metadata_mismatch(cache: dict[str, Any], data_path: Path) -> None:
    metadata = cache.get("metadata")
    if not isinstance(metadata, dict):
        print("Warning: cache has no metadata to verify.", file=sys.stderr)
        return

    expected_size = metadata.get("data_size")
    expected_mtime_ns = metadata.get("data_mtime_ns")
    try:
        data_stat = data_path.stat()
    except OSError as exc:
        print(f"Warning: could not stat dataset {data_path}: {exc}", file=sys.stderr)
        return

    if expected_size is not None and expected_size != data_stat.st_size:
        print(
            "Warning: dataset size differs from skipped-index cache metadata.",
            file=sys.stderr,
        )
    if expected_mtime_ns is not None and expected_mtime_ns != data_stat.st_mtime_ns:
        print(
            "Warning: dataset mtime differs from skipped-index cache metadata.",
            file=sys.stderr,
        )


def iter_jsonl_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    record_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object record at {path}:{line_number}.")
            yield record_index, record
            record_index += 1


def iter_json_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object record at {path} index {index}.")
        yield index, record


def iter_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return iter_jsonl_records(path)
    return iter_json_records(path)


def build_output_record(
    index: int,
    record: dict[str, Any],
    image_folder: Path,
    records_only: bool,
) -> dict[str, Any]:
    if records_only:
        return record

    image = record.get("image")
    resolved_image_path = image_folder / str(image) if image else None
    return {
        "skipped_index": index,
        "image": image,
        "resolved_image_path": str(resolved_image_path) if resolved_image_path else None,
        "image_exists": resolved_image_path.is_file() if resolved_image_path else False,
        "record": record,
    }


def write_skipped_records(
    data_path: Path,
    image_folder: Path,
    skipped_indices: list[int],
    output_path: Path,
    records_only: bool,
    limit: int | None,
) -> int:
    requested_indices = sorted(set(skipped_indices))
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        requested_indices = requested_indices[:limit]

    requested = set(requested_indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for index, record in iter_records(data_path):
            if index % 100000 == 0 and index:
                print(f"Scanned {index:,} records; wrote {written:,}.")
            if index not in requested:
                continue

            output_record = build_output_record(
                index=index,
                record=record,
                image_folder=image_folder,
                records_only=records_only,
            )
            if not first:
                handle.write(",\n")
            json.dump(output_record, handle, ensure_ascii=False)
            first = False
            written += 1

            if written >= len(requested):
                break
        handle.write("\n]\n")
    return written


def main() -> None:
    args = parse_args()
    cache = load_cache(args.skipped_indices)
    metadata = cache.get("metadata") if isinstance(cache.get("metadata"), dict) else {}
    data_path = path_from_cache_or_arg(args.data_path, metadata, "data_path")
    image_folder = path_from_cache_or_arg(args.image_folder, metadata, "image_folder")
    output_path = args.output_file or args.output_dir / DEFAULT_OUTPUT_NAME

    if not args.no_verify_cache:
        warn_if_cache_metadata_mismatch(cache, data_path)

    skipped_indices = cache["skipped_indices"]
    print(f"Loaded {len(skipped_indices):,} skipped index/indices from {args.skipped_indices}.")
    print(f"Extracting from {data_path}.")
    written = write_skipped_records(
        data_path=data_path,
        image_folder=image_folder,
        skipped_indices=skipped_indices,
        output_path=output_path,
        records_only=args.records_only,
        limit=args.limit,
    )
    print(f"Wrote {written:,} skipped record(s) to {output_path}.")


if __name__ == "__main__":
    main()
