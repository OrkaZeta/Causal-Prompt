import argparse
import fnmatch
import itertools
import logging
import os
import re
import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
}
MODEL_SPECS = {
    "t2v-1.3B": {
        "version": "wan21",
        "sizes": ("480*832", "832*480"),
        "default_size": "832*480",
        "frame_num": 81,
        "sample_steps": 50,
        "sample_shift": 5.0,
        "sample_guide_scale": 5.0,
    },
    "t2v-14B": {
        "version": "wan21",
        "sizes": ("720*1280", "1280*720", "480*832", "832*480"),
        "default_size": "1280*720",
        "frame_num": 81,
        "sample_steps": 50,
        "sample_shift": 5.0,
        "sample_guide_scale": 5.0,
    },
    "t2v-5B": {
        "version": "wan22",
        "sizes": ("704*1280", "1280*704"),
        "default_size": "1280*704",
        "frame_num": 81,
        "sample_steps": 50,
        "sample_shift": 5.0,
        "sample_guide_scale": 5.0,
    },
    "t2v-A14B": {
        "version": "wan22",
        "sizes": ("720*1280", "1280*720", "480*832", "832*480"),
        "default_size": "1280*720",
        "frame_num": 81,
        "sample_steps": 40,
        "sample_shift": 12.0,
        "sample_guide_scale": (3.0, 4.0),
    },
}
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def load_prompts(value: str | Sequence[str]) -> list[str]:
    """Normalize a prompt string, a sequence of prompts, or a prompt text file."""
    if isinstance(value, str):
        prompts = [value]
    elif isinstance(value, Sequence):
        prompts = list(value)
    else:
        raise TypeError("prompt input must be a string or a sequence of strings")

    if not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError("every prompt must be a string")

    if len(prompts) == 1:
        prompt_file = Path(prompts[0]).expanduser()
        try:
            if prompt_file.is_file():
                prompts = prompt_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            # Long natural-language prompts are not filesystem paths.
            pass

    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError("prompt input contains no non-empty prompts")
    return prompts


def _jsonl_prompt_dataset(value, mode, split=None):
    values = [value] if isinstance(value, str) else list(value)
    if len(values) != 1 or Path(values[0]).suffix.lower() != ".jsonl":
        return None
    if __package__:
        from .prompt.schedule import JSONLPromptDataset
    else:
        from prompt.schedule import JSONLPromptDataset
    return JSONLPromptDataset(values[0], mode=mode, split=split)


def _validate_args(args: argparse.Namespace) -> None:
    if args.ckpt_dir is None:
        raise ValueError("Please specify the checkpoint directory with --ckpt_dir.")
    spec = MODEL_SPECS[args.task]
    if args.size is None:
        args.size = spec["default_size"]
    if args.size not in spec["sizes"]:
        supported = ", ".join(spec["sizes"])
        raise ValueError(
            f"Unsupported size {args.size} for task {args.task}; choose {supported}."
        )
    if spec["version"] == "wan22" and args.ring_size != 1:
        raise ValueError("Wan 2.2 supports Ulysses parallelism but not ring_size.")
    if spec["version"] == "wan21" and args.convert_model_dtype:
        raise ValueError("--convert_model_dtype is only supported by Wan 2.2.")

    for name in (
        "frame_num",
        "sample_steps",
        "sample_shift",
        "sample_guide_scale",
    ):
        if getattr(args, name) is None:
            setattr(args, name, spec[name])
    if args.frame_num <= 0 or args.frame_num % 4 != 1:
        raise ValueError("--frame_num must be a positive integer of the form 4n+1.")
    if getattr(args, "fps", 16.0) <= 0:
        raise ValueError("--fps must be positive.")
    if args.save_file is not None and Path(args.save_file).suffix.lower() != ".mp4":
        raise ValueError("--save_file must have an .mp4 extension.")

    raw_seeds = getattr(args, "seeds", None)
    if raw_seeds is None:
        raw_seeds = getattr(args, "base_seed", 0)
    if isinstance(raw_seeds, int):
        raw_seeds = [raw_seeds]
    args.seeds = list(dict.fromkeys(int(seed) for seed in raw_seeds))
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seed values must be non-negative integers.")
    if getattr(args, "max_samples", None) is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    args.base_seed = args.seeds[0]

    prompt_schedule = getattr(args, "prompt_schedule", "full")
    split = getattr(args, "split", "test")
    args.prompt_dataset = _jsonl_prompt_dataset(
        args.prompt, prompt_schedule, None if split == "all" else split
    )
    save_files = getattr(args, "save_files", None)
    if args.prompt_dataset is not None:
        args.prompts = None
        if args.save_file is not None or save_files is not None:
            raise ValueError("JSONL inference uses --save_dir, not save_file/save_files.")
    else:
        args.prompts = load_prompts(args.prompt)
    if save_files is not None and args.prompts is not None:
        if len(args.seeds) > 1:
            raise ValueError("save_files cannot be combined with multiple seeds.")
        if len(save_files) != len(args.prompts):
            raise ValueError("save_files must contain exactly one path per prompt.")
        if any(Path(path).suffix.lower() != ".mp4" for path in save_files):
            raise ValueError("Every save_files path must have an .mp4 extension.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save Wan text-to-video samples."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=[DEFAULT_PROMPT],
        help=(
            "A JSONL dataset scheduled lazily during inference, a quoted prompt, "
            "several quoted prompts, or a text file with one prompt per line."
        ),
    )
    parser.add_argument(
        "--prompt_schedule",
        choices=("full",),
        default="full",
        help="Full-video schedule applied lazily to each JSONL row.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=tuple(MODEL_SPECS),
        help=(
            "Text-to-video model: Wan 2.1 1.3B/14B, or Wan 2.2 "
            "text-only 5B/T2V A14B."
        ),
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        choices=list(SIZE_CONFIGS),
        help="Generated video size; defaults to the selected model's native size.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="Number of frames (4n+1); defaults to the selected model's value.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Output video frame rate (default: 16).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help=(
            "Output .mp4 path. For multiple prompts, an index is appended to the "
            "filename."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory; JSONL videos use <prompt_id>_seed<seed>.mp4.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Do not regenerate output videos that already exist.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Offload the model to CPU after each model forward.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses parallelism size in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="Ring-attention parallelism size in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp", action="store_true", help="Use FSDP for T5."
    )
    parser.add_argument(
        "--t5_cpu", action="store_true", help="Place the T5 model on CPU."
    )
    parser.add_argument(
        "--dit_fsdp", action="store_true", help="Use FSDP for DiT."
    )
    parser.add_argument(
        "--seed",
        "--base_seed",
        dest="seeds",
        type=int,
        nargs="+",
        default=[0],
        help="One or more seeds generated for every prompt (default: 0).",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=("unipc", "dpm++"),
        help="Sampling solver.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Sampling steps; defaults to the selected model's value.",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Flow-matching shift; defaults to the selected model's value.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Guidance scale; defaults to the selected model's value.",
    )
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        help="Convert Wan 2.2 model parameters to its configured dtype.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the Wan diffusion-model forward pass with torch.compile.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode (default: default).",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--sample-id-glob", action="append", default=[])

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _compile_pipeline_models(pipeline, mode: str) -> tuple[str, ...]:
    """Compile only DiT forwards while preserving model movement/offloading."""
    import torch

    compiled = []
    for name in ("model", "high_noise_model", "low_noise_model"):
        model = getattr(pipeline, name, None)
        if model is None:
            continue
        model.forward = torch.compile(model.forward, mode=mode, dynamic=False)
        compiled.append(name)
    if not compiled:
        raise AttributeError("Wan pipeline exposes no diffusion model to compile.")
    return tuple(compiled)


def _output_path(
    args: argparse.Namespace,
    prompt: str,
    index: int,
    prompt_count: int | None,
    prompt_id: str | None = None,
) -> Path:
    save_dir = getattr(args, "save_dir", None)
    if prompt_id is not None:
        output_dir = Path(save_dir or ".").expanduser()
        return output_dir / f"{prompt_id}_seed{args.base_seed}.mp4"
    save_files = getattr(args, "save_files", None)
    if save_files is not None:
        return Path(save_files[index]).expanduser()
    if args.save_file is not None:
        path = Path(args.save_file).expanduser()
        if prompt_count == 1 and len(args.seeds) == 1:
            return path
        assert prompt_count is not None
        width = max(4, len(str(prompt_count - 1)))
        seed_part = f"_seed{args.base_seed}" if len(args.seeds) > 1 else ""
        return path.with_name(
            f"{path.stem}_{index:0{width}d}{seed_part}{path.suffix}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = re.sub(r"[^\w.-]+", "_", prompt, flags=re.UNICODE).strip("_")
    prompt_slug = prompt_slug[:50] or "prompt"
    size = args.size.replace("*", "x") if sys.platform == "win32" else args.size
    index_part = f"_{index:04d}" if prompt_count is not None and prompt_count > 1 else ""
    path = Path(
        f"{args.task}_{size}_{args.ulysses_size}_{args.ring_size}"
        f"{index_part}_{prompt_slug}_seed{args.base_seed}_{timestamp}.mp4"
    )
    return Path(save_dir).expanduser() / path if save_dir is not None else path


def generate(args: argparse.Namespace) -> list[Path]:
    """Generate every requested seed per prompt and return all output paths."""
    if not hasattr(args, "prompts"):
        _validate_args(args)

    import torch
    import torch.distributed as dist

    spec = MODEL_SPECS[args.task]
    if spec["version"] == "wan21" and __package__:
        from .models import wan21
        from .models.wan21.configs import WAN_CONFIGS
        from .models.wan21.utils.utils import cache_video as save_video
    elif spec["version"] == "wan21":
        from models import wan21
        from models.wan21.configs import WAN_CONFIGS
        from models.wan21.utils.utils import cache_video as save_video
    elif __package__:
        from .models import wan22
        from .models.wan22.configs import WAN_CONFIGS
        from .models.wan22.distributed.util import init_distributed_group
        from .models.wan22.utils.utils import save_video
    else:
        from models import wan22
        from models.wan22.configs import WAN_CONFIGS
        from models.wan22.distributed.util import init_distributed_group
        from models.wan22.utils.utils import save_video

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = world_size == 1
        logging.info("offload_model was not specified; using %s.", args.offload_model)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("T5/DiT FSDP requires a distributed environment.")
        if args.ulysses_size > 1 or args.ring_size > 1:
            raise ValueError("Context parallelism requires a distributed environment.")

    use_parallel = args.ulysses_size > 1 or args.ring_size > 1
    if spec["version"] == "wan21" and use_parallel:
        if args.ulysses_size * args.ring_size != world_size:
            raise ValueError(
                "For Wan 2.1, ulysses_size * ring_size must equal WORLD_SIZE."
            )
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
    elif spec["version"] == "wan22" and args.ulysses_size > 1:
        if args.ulysses_size != world_size:
            raise ValueError("For Wan 2.2, ulysses_size must equal WORLD_SIZE.")
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError(
            f"cfg.num_heads ({cfg.num_heads}) must be divisible by ulysses_size "
            f"({args.ulysses_size})."
        )

    if dist.is_initialized():
        seeds = [args.seeds] if rank == 0 else [None]
        dist.broadcast_object_list(seeds, src=0)
        args.seeds = seeds[0]

    logging.info("Generation job args: %s", args)
    if spec["version"] == "wan21":
        logging.info("Creating Wan 2.1 WanT2V pipeline.")
        pipeline = wan21.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=use_parallel,
            t5_cpu=args.t5_cpu,
        )
    elif args.task == "t2v-5B":
        logging.info("Creating Wan 2.2 5B text-to-video pipeline.")
        pipeline = wan22.WanT2V5B(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
    else:
        logging.info("Creating Wan 2.2 WanT2V pipeline.")
        pipeline = wan22.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

    if getattr(args, "torch_compile", False):
        compile_mode = getattr(args, "torch_compile_mode", "default")
        compiled = _compile_pipeline_models(pipeline, compile_mode)
        logging.info(
            "torch.compile enabled for %s (mode=%s, dynamic=False).",
            ", ".join(compiled),
            compile_mode,
        )

    if args.prompt_dataset is not None:
        sample_id_globs = getattr(args, "sample_id_glob", [])
        prompt_items = itertools.islice(
            ((item.prompt_id, item.prompt) for item in args.prompt_dataset
             if not sample_id_globs
             or any(fnmatch.fnmatchcase(item.prompt_id, pattern)
                    for pattern in sample_id_globs)),
            args.max_samples,
        )
        prompt_count = None
    else:
        prompt_items = ((None, prompt) for prompt in args.prompts)
        prompt_count = len(args.prompts)

    output_paths = []
    for index, (prompt_id, prompt) in enumerate(prompt_items):
        for seed in args.seeds:
            args.base_seed = seed
            output_path = _output_path(
                args,
                prompt,
                index,
                prompt_count,
                prompt_id=prompt_id,
            )
            skip_video = (
                rank == 0
                and getattr(args, "skip_existing", False)
                and output_path.is_file()
            )
            if dist.is_initialized():
                skip_state = [skip_video] if rank == 0 else [None]
                dist.broadcast_object_list(skip_state, src=0)
                skip_video = skip_state[0]
            if skip_video:
                logging.info("Skipping existing video: %s", output_path)
                if rank == 0:
                    output_paths.append(output_path)
                continue

            progress = (
                str(index + 1)
                if prompt_count is None
                else f"{index + 1}/{prompt_count}"
            )
            logging.info(
                "Generating video %s (id=%s) with seed %d.",
                progress,
                prompt_id or "direct-input",
                seed,
            )
            logging.debug("Scheduled prompt: %s", prompt)
            generate_kwargs = {
                "size": SIZE_CONFIGS[args.size],
                "frame_num": args.frame_num,
                "shift": args.sample_shift,
                "sample_solver": args.sample_solver,
                "sampling_steps": args.sample_steps,
                "guide_scale": args.sample_guide_scale,
                "seed": seed,
                "offload_model": args.offload_model,
            }
            video = pipeline.generate(prompt, **generate_kwargs)

            if rank == 0:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info("Saving generated video to %s", output_path)
                save_video(
                    tensor=video[None],
                    save_file=str(output_path),
                    fps=getattr(args, "fps", 16.0),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                output_paths.append(output_path)
            del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")
    return output_paths


if __name__ == "__main__":
    generate(_parse_args())
