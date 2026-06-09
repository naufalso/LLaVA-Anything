#!/usr/bin/env python
"""Check saved model and DeepSpeed checkpoint shards for non-finite tensors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _iter_tensors(obj: Any, prefix: str = ""):
    if torch.is_tensor(obj):
        yield prefix or "<root>", obj
        return
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(value, child)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for index, value in enumerate(obj):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_tensors(value, child)


def _check_tensors(path: Path, max_bad: int) -> tuple[int, int, list[str]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    tensor_count = 0
    element_count = 0
    bad: list[str] = []
    for name, tensor in _iter_tensors(data):
        tensor_count += 1
        element_count += tensor.numel()
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            finite = torch.isfinite(tensor)
            if not bool(finite.all()):
                nonfinite = int((~finite).sum().item())
                bad.append(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} nonfinite={nonfinite}")
                if len(bad) >= max_bad:
                    break
    return tensor_count, element_count, bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-bad", type=int, default=20)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint}")

    shard_paths = sorted((checkpoint / "global_step200").glob("*.pt"))
    if not shard_paths:
        shard_paths = sorted(checkpoint.glob("**/*.pt"))
    if not shard_paths:
        raise SystemExit(f"No .pt checkpoint shards found under {checkpoint}")

    total_bad = 0
    for path in shard_paths:
        print(f"[CHECK] {path}", flush=True)
        tensor_count, element_count, bad = _check_tensors(path, args.max_bad)
        print(
            f"[RESULT] tensors={tensor_count} elements={element_count} bad_tensors={len(bad)}",
            flush=True,
        )
        for item in bad:
            print(f"[BAD] {item}", flush=True)
        total_bad += len(bad)
    print(f"[SUMMARY] files={len(shard_paths)} bad_tensors={total_bad}", flush=True)


if __name__ == "__main__":
    main()
