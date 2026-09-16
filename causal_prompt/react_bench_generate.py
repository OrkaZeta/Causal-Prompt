"""Generate ReactBench full-prompt reference videos with a Wan T2V model."""
from __future__ import annotations

import argparse
import json
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from .wan_generate import generate


DEFAULT_INPUT = Path("data/react_bench_test.jsonl")
DEFAULT_OUTPUT = Path("outputs/react_bench/test/gt_video")
DEFAULT_CHECKPOINT = Path("/projects/hi-paris/ZiyiData/Models/Wan2.1-T2V-14B")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ReactBenchItem:
    sample_id: str
    prompt: str


def load_items(path: Path) -> list[ReactBenchItem]:
    items: list[ReactBenchItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            sample_id = record.get("ID")
            prompt = record.get("Full Prompt")
            if not isinstance(sample_id, str) or SAFE_ID.fullmatch(sample_id) is None:
                raise ValueError(f"Invalid ID at {path}:{line_number}: {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate ID at {path}:{line_number}: {sample_id}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing Full Prompt at {path}:{line_number}")
            seen.add(sample_id)
            items.append(ReactBenchItem(sample_id, prompt.strip()))
    if not items:
        raise ValueError(f"No ReactBench records found in {path}")
    return items


def select_shard(
    items: list[ReactBenchItem], shard_index: int, num_shards: int
) -> list[ReactBenchItem]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return items[shard_index::num_shards]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--task", choices=("t2v-1.3B", "t2v-14B", "t2v-5B"), default="t2v-14B"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    selected = select_shard(load_items(cli.input), cli.shard_index, cli.num_shards)
    outputs = [cli.output_dir / f"{item.sample_id}.mp4" for item in selected]
    print(
        f"ReactBench shard {cli.shard_index}/{cli.num_shards}: "
        f"{len(selected)} videos -> {cli.output_dir}",
        flush=True,
    )
    if cli.validate_only:
        return

    wan_args = Namespace(
        prompt=[item.prompt for item in selected],
        prompt_schedule="full",
        task=cli.task,
        size="1280*720",
        frame_num=81,
        fps=16.0,
        ckpt_dir=str(cli.checkpoint),
        save_file=None,
        save_files=[str(path) for path in outputs],
        save_dir=None,
        skip_existing=not cli.overwrite,
        offload_model=False,
        ulysses_size=1,
        ring_size=1,
        t5_fsdp=False,
        t5_cpu=False,
        dit_fsdp=False,
        seeds=[cli.seed],
        sample_solver="unipc",
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        convert_model_dtype=False,
        split="all",
        max_samples=None,
        sample_id_glob=[],
        torch_compile=cli.torch_compile,
        torch_compile_mode=cli.torch_compile_mode,
    )
    generate(wan_args)


if __name__ == "__main__":
    main()
