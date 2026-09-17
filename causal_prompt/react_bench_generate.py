"""Generate ReactBench videos with Wan2.2-Lightning and TS-Attn."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TS_ATTN_SOURCE = REPO_ROOT / "third_party" / "TS-Attn"
DEFAULT_INPUT = REPO_ROOT / "data" / "react_bench_test.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "react_bench" / "test" / "gt_video"
DEFAULT_BASE_MODEL = Path(
    "/projects/hi-paris/ZiyiData/Models/Wan2.2-T2V-A14B-Diffusers"
)
DEFAULT_LIGHTNING_LORA = Path(
    "/projects/hi-paris/ZiyiData/Models/Wan2.2-Lightning/"
    "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"
)
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SUBJECT_BOUNDARY = re.compile(
    r"\s+(?:is|are|has|have|stands?|sits?|lies?|remains?)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ReactBenchItem:
    sample_id: str
    prompt: str
    events: tuple[str, ...]
    subject: str


def _subject_from_anchor(anchor: str) -> str:
    """Return a literal anchor phrase that is guaranteed to occur in the prompt."""
    first_sentence = anchor.split(".", 1)[0].strip()
    subject = SUBJECT_BOUNDARY.split(first_sentence, maxsplit=1)[0].strip()
    subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
    if not subject:
        raise ValueError(f"Cannot infer TS-Attn subject from Anchor: {anchor!r}")
    return subject


def _event_ranges(count: int) -> list[float]:
    """Split the video evenly while making floating-point values sum exactly to 1."""
    if count <= 0:
        raise ValueError("TS-Attn requires at least one event")
    ranges = [1.0 / count] * count
    ranges[-1] = 1.0 - sum(ranges[:-1])
    return ranges


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
            anchor = record.get("Anchor")
            if not isinstance(sample_id, str) or SAFE_ID.fullmatch(sample_id) is None:
                raise ValueError(f"Invalid ID at {path}:{line_number}: {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate ID at {path}:{line_number}: {sample_id}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing Full Prompt at {path}:{line_number}")
            if not isinstance(anchor, str) or not anchor.strip():
                raise ValueError(f"Missing Anchor at {path}:{line_number}")
            events = tuple(
                value.strip()
                for key in ("E1", "E2", "E3")
                if isinstance((value := record.get(key)), str) and value.strip()
            )
            if len(events) != record.get("Event Count"):
                raise ValueError(
                    f"Event Count mismatch at {path}:{line_number}: "
                    f"declared={record.get('Event Count')!r}, found={len(events)}"
                )
            missing_events = [event for event in events if event not in prompt]
            if missing_events:
                raise ValueError(
                    f"TS-Attn events must occur literally in Full Prompt at "
                    f"{path}:{line_number}: {missing_events!r}"
                )
            subject = _subject_from_anchor(anchor)
            if subject not in prompt:
                raise ValueError(
                    f"Inferred TS-Attn subject is absent from Full Prompt at "
                    f"{path}:{line_number}: {subject!r}"
                )
            seen.add(sample_id)
            items.append(ReactBenchItem(sample_id, prompt.strip(), events, subject))
    if not items:
        raise ValueError(f"No ReactBench records found in {path}")
    return items


def _require_runtime(base_model: Path, lightning_lora: Path) -> None:
    required = (
        TS_ATTN_SOURCE / "models_2_2_t2v" / "pipeline_TsAttn.py",
        TS_ATTN_SOURCE / "models_2_2_t2v" / "transformer_TsAttn.py",
        base_model / "model_index.json",
        base_model / "transformer" / "config.json",
        base_model / "transformer_2" / "config.json",
        base_model / "vae" / "config.json",
        lightning_lora / "high_noise_model.safetensors",
        lightning_lora / "low_noise_model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Wan2.2-Lightning + TS-Attn runtime is incomplete; missing:\n  "
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


def _load_lightning_lora(transformer, path: Path, adapter_name: str) -> None:
    """Load a native LightX2V Wan LoRA into one Diffusers transformer."""
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_wan_lora_to_diffusers,
    )
    from safetensors.torch import load_file

    # Preserve alpha: the V2.0 checkpoint uses alpha=8 with rank=64, so its
    # effective adapter scale is 0.125 rather than 1.0.
    native = load_file(str(path), device="cpu")
    converted = _convert_non_diffusers_wan_lora_to_diffusers(native)
    transformer.load_lora_adapter(converted, adapter_name=adapter_name)
    transformer.set_adapters([adapter_name], weights=[1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--prompt",
        help="Generate one literal prompt instead of reading the JSONL input.",
    )
    parser.add_argument(
        "--output-id",
        help="Filename ID for --prompt; output is <output-dir>/<output-id>.mp4.",
    )
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="TS-Attn motion phrase for --prompt; repeat in temporal order.",
    )
    parser.add_argument(
        "--subject",
        help="TS-Attn subject phrase for --prompt; it must occur in the prompt.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lightning-lora", type=Path, default=DEFAULT_LIGHTNING_LORA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--ts-attn-ratio",
        type=float,
        default=0.25,
        help="Fraction of four denoising steps controlled by TS-Attn (default: one step).",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0.25 <= args.ts_attn_ratio <= 1.0:
        raise ValueError("--ts-attn-ratio must be in [0.25, 1.0] for four-step inference")

    if args.prompt is not None:
        prompt = args.prompt.strip()
        events = tuple(event.strip() for event in args.event if event.strip())
        subject = args.subject.strip() if args.subject else ""
        if not prompt:
            raise ValueError("--prompt must not be empty")
        if not isinstance(args.output_id, str) or SAFE_ID.fullmatch(args.output_id) is None:
            raise ValueError("--prompt requires a filename-safe --output-id")
        if not events or any(event not in prompt for event in events):
            raise ValueError("--prompt requires literal, in-prompt --event values")
        if not subject or subject not in prompt:
            raise ValueError("--prompt requires an in-prompt --subject value")
        if args.max_samples is not None:
            raise ValueError("--max-samples cannot be combined with --prompt")
        items = [ReactBenchItem(args.output_id, prompt, events, subject)]
    else:
        if args.output_id is not None or args.event or args.subject is not None:
            raise ValueError("--output-id, --event and --subject require --prompt")
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
    sys.path.insert(0, str(TS_ATTN_SOURCE))

    import torch
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_to_video
    from models_2_2_t2v.pipeline_TsAttn import WanTsAttnPipeline
    from models_2_2_t2v.transformer_TsAttn import WanTsAttnTransformer3DModel

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    logging.info(
        "Loading Wan2.2-Lightning + TS-Attn: 832x480, 81 frames, 16 fps, "
        "4 Euler steps, CFG disabled, TS ratio=%.2f, seed=%d.",
        args.ts_attn_ratio,
        args.seed,
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTsAttnTransformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer_2 = WanTsAttnTransformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer_2", torch_dtype=torch.bfloat16
    )
    _load_lightning_lora(
        transformer,
        args.lightning_lora / "high_noise_model.safetensors",
        "lightning_high_noise",
    )
    _load_lightning_lora(
        transformer_2,
        args.lightning_lora / "low_noise_model.safetensors",
        "lightning_low_noise",
    )
    pipeline = WanTsAttnPipeline.from_pretrained(
        args.base_model,
        vae=vae,
        transformer=transformer,
        transformer_2=transformer_2,
        torch_dtype=torch.bfloat16,
    )

    class LightningEulerScheduler(FlowMatchEulerDiscreteScheduler):
        """Match LightX2V's shifted four-step Euler schedule exactly."""

        def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
            if num_inference_steps is None:
                return super().set_timesteps(device=device, **kwargs)
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            sigmas = 5.0 * sigmas / (1.0 + 4.0 * sigmas)
            return super().set_timesteps(device=device, sigmas=sigmas.tolist())

    pipeline.scheduler = LightningEulerScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    pipeline.to("cuda")
    if args.compile:
        logging.info("Compiling both TS-Attn transformers with torch.compile.")
        pipeline.transformer.compile(mode="max-autotune-no-cudagraphs")
        pipeline.transformer_2.compile(mode="max-autotune-no-cudagraphs")

    model_config = {
        "Env_args": {
            "time_step_ratio": args.ts_attn_ratio,
            "mask_type": "soft",
            "threshold": 0.2,
            "use_erode": True,
        }
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(pending, 1):
        output = args.output_dir / f"{item.sample_id}.mp4"
        logging.info(
            "Generating %d/%d: %s; subject=%r; events=%r",
            index,
            len(pending),
            item.sample_id,
            item.subject,
            item.events,
        )
        frames = pipeline(
            prompt=item.prompt,
            event_list=list(item.events),
            event_range=_event_ranges(len(item.events)),
            subject=[item.subject],
            model_configs=model_config,
            negative_prompt=None,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]
        export_to_video(frames, str(output), fps=16)
        logging.info("Saved %s", output)
        del frames

    torch.cuda.synchronize()
    logging.info("Finished %d videos.", len(pending))


if __name__ == "__main__":
    main()
