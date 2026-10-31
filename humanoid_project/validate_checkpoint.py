"""Fail if an RSL-RL checkpoint is missing, tiny, unreadable, or non-finite."""

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()
path = Path(args.checkpoint)
if not path.is_file() or path.stat().st_size < 100_000:
    raise SystemExit(f"Checkpoint is missing or too small: {path}")

payload = torch.load(path, map_location="cpu", weights_only=False)
tensor_count = 0
non_finite = []


def visit(value, prefix="root"):
    global tensor_count
    if torch.is_tensor(value):
        tensor_count += 1
        if value.is_floating_point() and not torch.isfinite(value).all():
            non_finite.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            visit(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            visit(child, f"{prefix}[{index}]")


visit(payload)
if tensor_count == 0:
    raise SystemExit("Checkpoint contains no tensors")
if non_finite:
    raise SystemExit("Checkpoint contains non-finite tensors: " + ", ".join(non_finite[:10]))
print(f"Checkpoint valid: {path} ({tensor_count} tensors)")
