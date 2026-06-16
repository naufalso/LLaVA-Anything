#!/usr/bin/env python3
"""Merge dataset-analysis broken images into training skipped-image caches.

``broken_images.csv`` uses raw dataset record indices.  The training
``skipped_image_indices.json`` cache uses indices after optional
``data.require_image`` filtering.  This script translates those coordinates
before merging so the cache continues to point at the records training sees.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INDEX_COLUMN = "record_index"


@dataclass(frozen=True)
class MappingSpec:
    data_path: Path
    max_samples: int | None
    require_image: bool


@dataclass
class BrokenIndexSet:
    indices: set[int]
    rows_read: int
    rows_used: int
    reasons: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge indices from dataset_analysis/broken_images.csv into one or "
            "more skipped_image_indices.json files."
        )
    )
    parser.add_argument(
        "broken_images_csv",
        type=Path,
        help="Path to dataset_analysis/broken_images.csv.",
    )
    parser.add_argument(
        "skipped_indices_json",
        nargs="+",
        type=Path,
        help="One or more skipped_image_indices.json cache files to update.",
    )
    parser.add_argument(
        "--index-column",
        default=DEFAULT_INDEX_COLUMN,
        help=f"CSV column containing record indices. Default: {DEFAULT_INDEX_COLUMN}",
    )
    parser.add_argument(
        "--coordinate",
        choices=("raw", "cache"),
        default="raw",
        help=(
            "Coordinate system for CSV indices. Use 'raw' for analyse_dataset.py "
            "outputs, or 'cache' if the indices already match the training cache. "
            "Default: raw"
        ),
    )
    parser.add_argument(
        "--reason",
        action="append",
        help=(
            "Only use CSV rows with this exact skip_reason. Can be repeated. "
            "Default: use every row in the CSV."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Dataset path used for raw-to-cache mapping. Defaults to "
            "metadata.data_path from each cache."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Dataset max_samples used for raw-to-cache mapping. Defaults to "
            "metadata.max_samples from each cache."
        ),
    )
    require_group = parser.add_mutually_exclusive_group()
    require_group.add_argument(
        "--require-image",
        dest="require_image",
        action="store_true",
        help="Map raw indices after dropping text-only records.",
    )
    require_group.add_argument(
        "--no-require-image",
        dest="require_image",
        action="store_false",
        help="Map raw indices without dropping text-only records.",
    )
    parser.set_defaults(require_image=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updated cache files. Without this flag the script is a dry run.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Before writing, copy each changed cache to <path>.bak.",
    )
    return parser.parse_args()


def load_broken_indices(
    path: Path,
    index_column: str,
    allowed_reasons: set[str] | None,
) -> BrokenIndexSet:
    indices: set[int] = set()
    reasons: Counter[str] = Counter()
    rows_read = 0
    rows_used = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or index_column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain CSV column {index_column!r}.")

        for row in reader:
            rows_read += 1
            reason = row.get("skip_reason", "")
            if allowed_reasons is not None and reason not in allowed_reasons:
                continue

            value = row.get(index_column, "")
            try:
                index = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {index_column} value at CSV row {rows_read + 1}: {value!r}"
                ) from exc
            if index < 0:
                raise ValueError(
                    f"Invalid negative {index_column} at CSV row {rows_read + 1}: {index}"
                )

            indices.add(index)
            reasons.update([reason])
            rows_used += 1

    return BrokenIndexSet(
        indices=indices,
        rows_read=rows_read,
        rows_used=rows_used,
        reasons=reasons,
    )


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
    metadata = cache.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path} contains non-object metadata.")
    return cache


def mapping_spec_from_cache(cache: dict[str, Any], args: argparse.Namespace) -> MappingSpec:
    metadata = cache.get("metadata") if isinstance(cache.get("metadata"), dict) else {}

    data_path = args.data_path
    if data_path is None:
        data_path_value = metadata.get("data_path")
        if not data_path_value:
            raise ValueError(
                "Cache metadata is missing data_path; pass --data-path for raw mapping."
            )
        data_path = Path(str(data_path_value))

    max_samples = args.max_samples
    if max_samples is None:
        metadata_max_samples = metadata.get("max_samples")
        if metadata_max_samples is not None:
            max_samples = int(metadata_max_samples)
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative.")

    require_image = args.require_image
    if require_image is None:
        require_image = bool(metadata.get("require_image", False))

    return MappingSpec(
        data_path=data_path,
        max_samples=max_samples,
        require_image=bool(require_image),
    )


def record_has_image(record: dict[str, Any]) -> bool:
    return bool(record.get("image"))


def iter_jsonl_records(path: Path, max_samples: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
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
                raise ValueError(f"Expected JSON object record at {path}:{line_number}.")
            yield yielded, record
            yielded += 1


def iter_json_records(path: Path, max_samples: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return iter_jsonl_records(path, max_samples)

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    def iterator() -> Iterable[tuple[int, dict[str, Any]]]:
        for index, record in enumerate(records):
            if max_samples is not None and index >= max_samples:
                break
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object record at {path} index {index}.")
            yield index, record

    return iterator()


def map_raw_indices_to_cache_indices(
    raw_indices: set[int],
    spec: MappingSpec,
) -> tuple[set[int], set[int]]:
    if not raw_indices:
        return set(), set()

    max_target = max(raw_indices)
    if spec.max_samples is not None:
        eligible_targets = {index for index in raw_indices if index < spec.max_samples}
        dropped_targets = set(raw_indices) - eligible_targets
    else:
        eligible_targets = set(raw_indices)
        dropped_targets = set()

    if not eligible_targets:
        return set(), set(raw_indices)

    target_lookup = eligible_targets
    mapped: set[int] = set()
    unmapped = set(dropped_targets)
    seen_targets: set[int] = set()
    cache_index = 0

    for raw_index, record in iter_json_records(spec.data_path, spec.max_samples):
        if raw_index and raw_index % 500000 == 0:
            print(f"  scanned {raw_index:,} raw records; mapped {len(mapped):,}.")

        include_in_cache = not spec.require_image or record_has_image(record)
        if raw_index in target_lookup:
            seen_targets.add(raw_index)
            if include_in_cache:
                mapped.add(cache_index)
            else:
                unmapped.add(raw_index)

            if raw_index >= max_target:
                break

        if include_in_cache:
            cache_index += 1

    missing_targets = eligible_targets - seen_targets
    unmapped.update(missing_targets)
    return mapped, unmapped


def validate_indices_in_range(
    indices: set[int],
    cache: dict[str, Any],
    cache_path: Path,
) -> None:
    metadata = cache.get("metadata") if isinstance(cache.get("metadata"), dict) else {}
    record_count = metadata.get("record_count")
    if record_count is None:
        return
    record_count = int(record_count)
    out_of_range = [index for index in indices if index < 0 or index >= record_count]
    if out_of_range:
        sample = sorted(out_of_range)[:10]
        raise ValueError(
            f"{cache_path} would receive {len(out_of_range)} out-of-range index/indices "
            f"for record_count={record_count}: {sample}"
        )


def write_cache(path: Path, cache: dict[str, Any], backup: bool) -> None:
    if backup:
        backup_path = path.with_name(f"{path.name}.bak")
        shutil.copy2(path, backup_path)
        print(f"  backup: {backup_path}")

    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key or '<empty>'}: {value}" for key, value in counter.most_common())


def main() -> None:
    args = parse_args()
    allowed_reasons = set(args.reason) if args.reason else None
    broken = load_broken_indices(
        args.broken_images_csv,
        index_column=args.index_column,
        allowed_reasons=allowed_reasons,
    )
    print(
        f"Loaded {broken.rows_used:,} usable broken-image row(s) "
        f"from {broken.rows_read:,} CSV row(s)."
    )
    print(f"Unique CSV indices: {len(broken.indices):,}")
    print(f"Reasons: {format_counter(broken.reasons)}")

    caches_by_path = {
        path: load_cache(path)
        for path in args.skipped_indices_json
    }

    mapped_by_spec: dict[MappingSpec, tuple[set[int], set[int]]] = {}
    if args.coordinate == "raw":
        for cache in caches_by_path.values():
            spec = mapping_spec_from_cache(cache, args)
            if spec not in mapped_by_spec:
                print(
                    "Mapping raw indices with "
                    f"data_path={spec.data_path}, "
                    f"max_samples={spec.max_samples}, "
                    f"require_image={spec.require_image}"
                )
                mapped_by_spec[spec] = map_raw_indices_to_cache_indices(
                    broken.indices,
                    spec,
                )

    changed_paths: list[Path] = []
    for path, cache in caches_by_path.items():
        if args.coordinate == "raw":
            spec = mapping_spec_from_cache(cache, args)
            incoming_indices, unmapped = mapped_by_spec[spec]
        else:
            incoming_indices = set(broken.indices)
            unmapped = set()

        validate_indices_in_range(incoming_indices, cache, path)

        existing_indices = set(cache["skipped_indices"])
        merged_indices = sorted(existing_indices | incoming_indices)
        added = len(set(merged_indices) - existing_indices)
        already_present = len(incoming_indices & existing_indices)

        print(f"\n{path}")
        print(f"  existing: {len(existing_indices):,}")
        print(f"  incoming after mapping: {len(incoming_indices):,}")
        print(f"  already present: {already_present:,}")
        print(f"  added: {added:,}")
        print(f"  new total: {len(merged_indices):,}")
        if unmapped:
            sample = sorted(unmapped)[:10]
            print(
                f"  unmapped raw CSV indices: {len(unmapped):,}; sample: {sample}",
                file=sys.stderr,
            )

        if added == 0:
            continue

        cache["skipped_indices"] = merged_indices
        changed_paths.append(path)
        if args.write:
            write_cache(path, cache, backup=args.backup)

    if args.write:
        print(f"\nUpdated {len(changed_paths):,} cache file(s).")
    else:
        print("\nDry run only. Re-run with --write to update changed cache files.")


if __name__ == "__main__":
    main()
