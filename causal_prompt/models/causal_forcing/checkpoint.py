"""Split, independently readable Exp-1 training checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 3
COMPONENTS = (
    "generator",
    "fake_score",
    "generator_optimizer",
    "critic_optimizer",
    "ema",
    "generator_ema",
    "rng_state",
)


def is_split_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "COMPLETE").is_file()


def checkpoint_step(path: Path) -> int:
    return int(path.name.removeprefix("step_"))


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in (run_dir / "checkpoints").glob("step_*"):
        if is_split_checkpoint(path) or (path / "model.pt").is_file():
            candidates.append(path)
    return max(candidates, key=checkpoint_step) if candidates else None


def load_component(checkpoint_dir: Path, name: str) -> Any:
    path = checkpoint_dir / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint component: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(checkpoint_dir: Path, metadata: dict[str, Any], names: list[str]) -> None:
    files = {}
    for name in names:
        path = checkpoint_dir / f"{name}.pt"
        files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {"format_version": FORMAT_VERSION, **metadata, "files": files}
    path = checkpoint_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (checkpoint_dir / "COMPLETE").touch()


def validate_manifest(checkpoint_dir: Path, verify_hashes: bool = True) -> dict[str, Any]:
    if not (checkpoint_dir / "COMPLETE").is_file():
        raise RuntimeError(f"Incomplete split checkpoint: {checkpoint_dir}")
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    if int(manifest["format_version"]) != FORMAT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_dir}")
    for filename, expected in manifest["files"].items():
        path = checkpoint_dir / filename
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Invalid checkpoint component: {path}")
        if verify_hashes and _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
    return manifest


def assert_identical(expected: Any, actual: Any, location: str = "root") -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise TypeError(f"{location}: expected tensor")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        equal_nan = expected.dtype.kind in "fc"
        if not isinstance(actual, np.ndarray) or not np.array_equal(
            actual, expected, equal_nan=equal_nan
        ):
            raise ValueError(f"{location}: NumPy array differs")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"{location}: dictionary keys differ")
        for key in expected:
            assert_identical(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError(f"{location}: sequence differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_identical(left, right, f"{location}[{index}]")
    elif expected != actual:
        raise ValueError(f"{location}: value differs")


def save_component(checkpoint_dir: Path, name: str, value: Any) -> None:
    temporary = checkpoint_dir / f"{name}.pt.tmp"
    torch.save(value, temporary)
    os.replace(temporary, checkpoint_dir / f"{name}.pt")
