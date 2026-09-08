"""Single-GPU, text-only inference for the bundled Causal Forcing models."""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from tqdm import tqdm

from causal_prompt.prompt.schedule import SCHEDULES, load_scheduled_prompts


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
CF_DIR = PACKAGE_DIR / "models" / "causal_forcing"
DEFAULT_BASE_MODEL = Path(
    "/projects/hi-paris/ZiyiData/Models/Wan2.1-T2V-1.3B"
)

CF_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "T1-FW": {
        "config": CF_DIR / "configs" / "ar_diffusion_tf_framewise.yaml",
        "checkpoint": Path(
            "/projects/hi-paris/ZiyiData/Models/Causal-Forcing/framewise/"
            "ar_diffusion.pt"
        ),
        "use_ema": False,
    },
    "S3-FW-CF4STEP": {
        "config": CF_DIR / "configs" / "causal_forcing_dmd_framewise.yaml",
        "checkpoint": Path(
            "/projects/hi-paris/ZiyiData/Models/Causal-Forcing/framewise/"
            "causal_forcing.pt"
        ),
        "use_ema": True,
    },
    "S3-FW-CFPP2STEP": {
        "config": CF_DIR / "configs" / "causal_forcing_dmd_framewise_2step.yaml",
        "checkpoint": Path(
            "/projects/hi-paris/ZiyiData/Models/Causal-Forcing/"
            "causal-forcing++/framewise-2step.pt"
        ),
        "use_ema": True,
    },
}


def _load_config(config_path: Path, base_model_path: Path | None):
    config = OmegaConf.merge(
        OmegaConf.load(CF_DIR / "configs" / "default_config.yaml"),
        OmegaConf.load(config_path),
    )
    if base_model_path is not None:
        config.base_model_path = str(base_model_path)
    return config


def _load_generator_checkpoint(
    pipeline: torch.nn.Module, checkpoint_path: Path, use_ema: bool
) -> None:
    if checkpoint_path.is_dir():
        component = "ema.pt" if use_ema else "generator.pt"
        component_path = checkpoint_path / component
        if not (checkpoint_path / "COMPLETE").is_file() or not component_path.is_file():
            raise RuntimeError(f"Incomplete split checkpoint: {checkpoint_path}")
        generator_state = torch.load(
            component_path, map_location="cpu", weights_only=False
        )
    else:
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state_key = "generator_ema" if use_ema else "generator"
        if state_key not in state_dict:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no {state_key!r} state dict."
            )
        generator_state = state_dict[state_key]
    try:
        pipeline.generator.load_state_dict(generator_state)
    except RuntimeError:
        fixed_state = {
            key.replace("model._fsdp_wrapped_module.", "model.", 1)
            if key.startswith("model._fsdp_wrapped_module.")
            else key: value
            for key, value in generator_state.items()
        }
        incompatible = pipeline.generator.load_state_dict(fixed_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logging.warning(
                f"Loaded {checkpoint_path} with missing keys="
                f"{incompatible.missing_keys} and unexpected keys="
                f"{incompatible.unexpected_keys}"
            )


def _reset_seed(seed: int) -> torch.Generator:
    """Reset every RNG and return a fresh CUDA generator for one video."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def generate(args: argparse.Namespace) -> list[Path]:
    """Generate every requested seed for each lazily scheduled JSONL record."""
    raw_seeds = getattr(args, "seeds", None)
    if raw_seeds is None:
        raw_seeds = getattr(args, "seed", 0)
    if isinstance(raw_seeds, int):
        raw_seeds = [raw_seeds]
    seeds = list(dict.fromkeys(int(seed) for seed in raw_seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Seed values must be non-negative integers.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if args.model not in CF_MODEL_SPECS:
        raise ValueError(f"Unknown Causal Forcing model: {args.model}")
    if args.prompt_schedule not in SCHEDULES:
        raise ValueError(f"Unknown prompt schedule: {args.prompt_schedule}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Causal Forcing inference requires one CUDA GPU.")

    spec = CF_MODEL_SPECS[args.model]
    config_path = Path(getattr(args, "config_path", None) or spec["config"])
    checkpoint_path = Path(
        getattr(args, "checkpoint_path", None) or spec["checkpoint"]
    )
    base_model_path = Path(
        getattr(args, "base_model_path", None) or DEFAULT_BASE_MODEL
    )
    for label, path, expected_directory in (
        ("config", config_path, False),
        ("checkpoint", checkpoint_path, False),
        ("base model", base_model_path, True),
    ):
        exists = path.is_dir() if expected_directory else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label.title()} does not exist: {path}")

    config = _load_config(config_path, base_model_path)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    # The copied Wan modules query the active CUDA device while importing, so
    # defer this import until after the single-GPU runtime has been validated.
    from causal_prompt.models.causal_forcing.pipeline import (
        CausalDiffusionInferencePipeline,
        CausalInferencePipeline,
    )

    pipeline_class = (
        CausalInferencePipeline
        if hasattr(config, "denoising_step_list")
        else CausalDiffusionInferencePipeline
    )
    pipeline = pipeline_class(config, device=device)
    use_ema_arg = getattr(args, "use_ema", None)
    use_ema = spec["use_ema"] if use_ema_arg is None else use_ema_arg
    _load_generator_checkpoint(
        pipeline,
        checkpoint_path,
        bool(use_ema),
    )
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    dataset_kind = getattr(args, "dataset_kind", None)
    if dataset_kind is None:
        dataset_kind = (
            "single_obj" if "single_obj" in Path(args.dataset).stem
            else "activitynet_causal"
        )
    output_dir = Path(args.save_dir) if args.save_dir else (
        PROJECT_DIR / "outputs" / "zeroshot_baseline" / dataset_kind
        / args.model / args.prompt_schedule / "vid"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_scheduled_prompts(
        args.dataset,
        mode=args.prompt_schedule,
        fps=16.0,
        temporal_downsample=4,
        chunk_size=int(config.num_frame_per_block),
        split=None if args.split == "all" else args.split,
    )

    _, configured_frames, channels, latent_height, latent_width = tuple(
        int(value) for value in config.image_or_video_shape
    )
    output_paths: list[Path] = []
    dataset_iterator = itertools.islice(iter(dataset), args.max_samples)
    batches = iter(lambda: list(itertools.islice(dataset_iterator, args.batch_size)), [])
    for batch in tqdm(batches, desc=f"{args.model}/{args.prompt_schedule}"):
        for seed in seeds:
            suffix = f"_seed{seed}"
            pending = [(item, output_dir / f"{item.prompt_id}{suffix}.mp4") for item in batch]
            existing = [path for _, path in pending if path.is_file()]
            if bool(getattr(args, "skip_existing", True)) and len(existing) == len(pending):
                output_paths.extend(existing)
                continue
            if bool(getattr(args, "skip_existing", True)) and existing:
                pending = [(item, path) for item, path in pending if not path.is_file()]
                output_paths.extend(existing)
            items = [item for item, _ in pending]
            if any(item.latent_frame_count != configured_frames for item in items):
                raise ValueError(f"Every sample must contain {configured_frames} latent frames.")

            random_generator = _reset_seed(seed)
            pipeline.kv_cache_pos = None
            noise = torch.randn(
                (
                    1,
                    configured_frames,
                    channels,
                    latent_height,
                    latent_width,
                ),
                generator=random_generator,
                device=device,
                dtype=torch.bfloat16,
            ).expand(len(items), -1, -1, -1, -1).clone()
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[item.prompt for item in items],
                block_text_prompts=None if args.prompt_schedule == "full" else
                    [list(item.chunk_prompts) for item in items],
                initial_latent=None,
                return_latents=True,
            )
            if any(video.shape[1] != item.pixel_frame_count for item in items):
                raise RuntimeError(
                    f"VAE decoded {video.shape[1]} frames; schedules require 81."
                )
            pixel_video = video.permute(0, 1, 3, 4, 2).mul(255).round().to(torch.uint8).cpu()
            for index, (_, output_path) in enumerate(pending):
                write_video(str(output_path), pixel_video[index], fps=16)
                output_paths.append(output_path)
            pipeline.vae.model.clear_cache()
            del video, latents, pixel_video, noise

    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scheduled text-to-video samples with Causal Forcing."
    )
    parser.add_argument("--model", choices=tuple(CF_MODEL_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt_schedule", choices=SCHEDULES, required=True)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--dataset_kind", choices=("single_obj", "activitynet_causal"))
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of prompts generated together on the GPU.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--base_model_path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, nargs="+", default=[0], dest="seeds")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.skip_existing = not args.overwrite
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generate(_parse_args())


if __name__ == "__main__":
    main()
