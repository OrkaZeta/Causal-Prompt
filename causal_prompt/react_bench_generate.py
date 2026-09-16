"""Generate ReactBench videos with the official Wan2.2-Lightning runtime."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTNING_SOURCE = REPO_ROOT / "third_party" / "LightX2V-Wan2.2-Lightning"
DEFAULT_INPUT = REPO_ROOT / "data" / "react_bench_test.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "react_bench" / "test" / "gt_video"
DEFAULT_BASE_MODEL = Path("/projects/hi-paris/ZiyiData/Models/Wan2.2-T2V-A14B")
DEFAULT_LIGHTNING_LORA = Path(
    "/projects/hi-paris/ZiyiData/Models/Wan2.2-Lightning/"
    "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"
)
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


def _require_runtime(base_model: Path, lightning_lora: Path) -> None:
    required = (
        LIGHTNING_SOURCE / "wan" / "__init__.py",
        base_model / "config.json",
        base_model / "models_t5_umt5-xxl-enc-bf16.pth",
        base_model / "Wan2.1_VAE.pth",
        base_model / "high_noise_model",
        base_model / "low_noise_model",
        lightning_lora / "high_noise_model.safetensors",
        lightning_lora / "low_noise_model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Wan2.2-Lightning runtime is incomplete; missing:\n  "
            + "\n  ".join(missing)
        )


def _is_complete_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            return (
                stream.width == 832
                and stream.height == 480
                and abs(fps - 16.0) < 0.01
                and stream.frames == 81
            )
    except (IndexError, OSError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lightning-lora", type=Path, default=DEFAULT_LIGHTNING_LORA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    items = load_items(args.input)
    if args.max_samples is not None:
        items = items[: args.max_samples]
    pending = [
        item
        for item in items
        if args.overwrite
        or not _is_complete_output(args.output_dir / f"{item.sample_id}.mp4")
    ]
    print(
        f"ReactBench: total={len(items)} pending={len(pending)} "
        f"output={args.output_dir}",
        flush=True,
    )
    if args.validate_only or not pending:
        return

    _require_runtime(args.base_model, args.lightning_lora)
    sys.path.insert(0, str(LIGHTNING_SOURCE))

    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    config = WAN_CONFIGS["t2v-A14B"]
    logging.info(
        "Loading Wan2.2-Lightning: 832x480, 81 frames, 16 fps, "
        "4 Euler steps, CFG disabled, seed=%d.",
        args.seed,
    )
    pipeline = wan.WanT2V(
        config=config,
        checkpoint_dir=str(args.base_model),
        lora_dir=str(args.lightning_lora),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(pending, 1):
        output = args.output_dir / f"{item.sample_id}.mp4"
        logging.info("Generating %d/%d: %s", index, len(pending), item.sample_id)
        video = pipeline.generate(
            item.prompt,
            size=SIZE_CONFIGS["832*480"],
            frame_num=81,
            shift=config.sample_shift,
            sample_solver="euler",
            sampling_steps=4,
            guide_scale=config.sample_guide_scale,
            seed=args.seed,
            offload_model=False,
        )
        save_video(
            tensor=video[None],
            save_file=str(output),
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info("Saved %s", output)
        del video

    torch.cuda.synchronize()
    logging.info("Finished %d videos.", len(pending))


if __name__ == "__main__":
    main()
