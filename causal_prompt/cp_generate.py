"""Generate videos from causal-prompt adapted checkpoints."""

from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path

import torch
from torchvision.io import write_video
from tqdm import tqdm

from causal_prompt.cf_generate import (
    PROJECT_DIR,
    _load_config,
    _load_generator_checkpoint,
    _reset_seed,
)
from causal_prompt.prompt.schedule import load_scheduled_prompts


DEFAULT_EXP_NAME = "exp1_cp_dmd"
STEP_PATTERN = re.compile(r"step_(\d{6})")


def discover_checkpoints(
    run_dir: Path, mode: str, steps: list[int] | None = None
) -> list[tuple[int, Path]]:
    root = run_dir / mode / "checkpoints"
    found = {}
    for directory in root.glob("step_*"):
        match = STEP_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        if (directory / "COMPLETE").is_file():
            found[int(match.group(1))] = directory
        elif (directory / "model.pt").is_file():
            found[int(match.group(1))] = directory / "model.pt"
    if not found:
        raise FileNotFoundError(f"No checkpoints found under {root}")
    requested = [max(found)] if not steps else list(dict.fromkeys(steps))
    missing = [step for step in requested if step not in found]
    if missing:
        raise FileNotFoundError(f"Missing {mode} checkpoints for steps: {missing}")
    return [(step, found[step]) for step in requested]


def _config_path(run_dir: Path, mode: str, override: Path | None) -> Path:
    """Use the exact resolved training config unless explicitly overridden."""
    if override is not None:
        return override
    return run_dir / mode / "resolved_config.yaml"


def generate(args: argparse.Namespace) -> list[Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("Causal Prompt inference requires one CUDA GPU.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seed values must be non-negative integers.")

    run_dir = Path(args.output_root) / args.exp_name / args.exp_id
    config_path = _config_path(run_dir, args.mode, args.config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Generation config does not exist: {config_path}")
    config = _load_config(config_path, args.base_model_path)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    from causal_prompt.models.causal_forcing.pipeline import (
        CausalDiffusionInferencePipeline,
        CausalInferencePipeline,
    )

    pipeline_class = (
        CausalInferencePipeline
        if hasattr(config, "denoising_step_list")
        else CausalDiffusionInferencePipeline
    )
    pipeline = pipeline_class(config, device=device).to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    _, expected_frames, channels, height, width = map(
        int, config.image_or_video_shape
    )

    outputs: list[Path] = []
    checkpoints = discover_checkpoints(run_dir, args.mode, args.steps)
    for step, checkpoint in checkpoints:
        logging.info(
            "Loading adapted exp=%s/%s mode=%s step=%d from %s",
            args.exp_name,
            args.exp_id,
            args.mode,
            step,
            checkpoint,
        )
        _load_generator_checkpoint(pipeline, checkpoint, use_ema=True)
        output_dir = run_dir / args.mode / "video" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_scheduled_prompts(
            args.dataset,
            mode=args.mode,
            fps=16.0,
            temporal_downsample=4,
            chunk_size=int(config.num_frame_per_block),
            split=None if args.split == "all" else args.split,
        )
        dataset_iterator = itertools.islice(iter(dataset), args.max_samples)
        batches = iter(
            lambda: list(itertools.islice(dataset_iterator, args.batch_size)), []
        )
        for batch in tqdm(
            batches, desc=f"{args.exp_id}/{args.mode}/step_{step:06d}"
        ):
            for seed in seeds:
                pending = [
                    (
                        item,
                        output_dir / f"{item.prompt_id}_seed{seed}.mp4",
                    )
                    for item in batch
                ]
                existing = [path for _, path in pending if path.is_file()]
                if not args.overwrite and len(existing) == len(pending):
                    outputs.extend(existing)
                    continue
                if not args.overwrite and existing:
                    pending = [
                        (item, path)
                        for item, path in pending
                        if not path.is_file()
                    ]
                    outputs.extend(existing)
                items = [item for item, _ in pending]
                if any(
                    item.latent_frame_count != expected_frames for item in items
                ):
                    raise ValueError(
                        f"Every sample must contain {expected_frames} latent frames."
                    )

                pipeline.kv_cache_pos = None
                noise = torch.randn(
                    (1, expected_frames, channels, height, width),
                    generator=_reset_seed(seed),
                    device=device,
                    dtype=torch.bfloat16,
                ).expand(len(items), -1, -1, -1, -1).clone()
                video, latents = pipeline.inference(
                    noise=noise,
                    text_prompts=[item.prompt for item in items],
                    block_text_prompts=[
                        list(item.chunk_prompts) for item in items
                    ],
                    initial_latent=None,
                    return_latents=True,
                )
                if any(
                    video.shape[1] != item.pixel_frame_count for item in items
                ):
                    raise RuntimeError(
                        f"VAE decoded {video.shape[1]} frames; schedules require "
                        f"{items[0].pixel_frame_count}."
                    )
                pixels = (
                    video.permute(0, 1, 3, 4, 2)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )
                for index, (_, output) in enumerate(pending):
                    write_video(str(output), pixels[index], fps=16)
                    outputs.append(output)
                pipeline.vae.model.clear_cache()
                del video, latents, pixels, noise
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from causal-prompt adapted checkpoints."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("causal", "current"), required=True)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Omit to use the largest completed checkpoint step.",
    )
    parser.add_argument(
        "--output_root", type=Path, default=PROJECT_DIR / "outputs"
    )
    parser.add_argument(
        "--split", choices=("train", "test", "all"), default="test"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, nargs="+", default=[0], dest="seeds")
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Override <run>/<mode>/resolved_config.yaml.",
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        help="Override base_model_path stored in the resolved training config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generate(_parse_args())


if __name__ == "__main__":
    main()
