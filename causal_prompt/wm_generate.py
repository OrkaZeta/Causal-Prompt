"""WM-LongLive / WM-AlayaWorld adapters for the zero-shot experiment layout."""
from __future__ import annotations
import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from causal_prompt.prompt.schedule import schedule_record, _prompt_for_interval

MODELS_ROOT = Path("/projects/hi-paris/ZiyiData/Models")


def records(args):
    selected = []
    seen = set()
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sid = record["sample_id"]
        if args.split != "all" and record["split"] != args.split:
            continue
        if args.sample_id_glob and not any(fnmatch.fnmatchcase(sid, p) for p in args.sample_id_glob):
            continue
        schedule_record(record, args.prompt_schedule)  # filename and timeline validation
        if sid in seen:
            raise ValueError(f"Duplicate sample ID: {sid}")
        seen.add(sid); selected.append(record)
    if args.max_samples is not None:
        selected = selected[:args.max_samples]
    if not selected:
        raise ValueError("No dataset records selected")
    return selected


def save_video(frames, fps, target):
    from torchvision.io import write_video
    # Preserve seconds, not native frame numbers; all report videos are 81 frames @ 16 fps.
    native = target.with_suffix(".native.mp4")
    temporary = target.with_suffix(".partial.mp4")
    write_video(str(native), frames, fps=fps, video_codec="libx264", options={"crf": "18"})
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(native),
                    "-vf", "fps=16", "-frames:v", "81", "-an", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(temporary)], check=True)
    os.replace(temporary, target)
    native.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["WM-LongLive", "WM-AlayaWorld"], required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--prompt_schedule", choices=["current", "causal"], required=True)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
    p.add_argument("--seed", nargs="+", type=int, default=[0])
    p.add_argument("--max_samples", type=int)
    p.add_argument("--sample-id-glob", action="append", default=[])
    p.add_argument("--checkpoint_path", type=Path)
    p.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    p.add_argument("--base_model_path", type=Path)
    p.add_argument("--image-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    selected = records(args)
    if any(s < 0 for s in args.seed):
        raise ValueError("Seeds must be non-negative")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, s, args.save_dir / f"{r['sample_id']}_seed{s}.mp4") for r in selected for s in args.seed]
    jobs = [(r, s, out) for r, s, out in jobs if args.overwrite or not (out.exists() and out.with_suffix('.json').exists())]
    if not jobs:
        print("All requested videos and metadata already exist."); return
    if args.model == "WM-LongLive":
        from causal_prompt.models.longlive.inference import LongLivePipeline
        checkpoint = args.checkpoint_path or args.models_root / "LongLive-2.0-5B"
        pipe = LongLivePipeline(checkpoint, args.base_model_path or args.models_root / "Wan2.2-TI2V-5B")
    else:
        if args.image_dir is None:
            raise ValueError("WM-AlayaWorld requires --image-dir with <sample_id>_seed<seed>.png")
        for record, seed, _ in jobs:
            image = args.image_dir / f"{record['sample_id']}_seed{seed}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
        from causal_prompt.models.alayaworld.inference import AlayaWorldPipeline
        checkpoint = args.checkpoint_path or args.models_root / "AlayaWorld-v1.1-stage3"
        pipe = AlayaWorldPipeline(checkpoint, args.models_root)
    for record, seed, out in jobs:
        scheduled = schedule_record(record, args.prompt_schedule, fps=pipe.fps,
                                    temporal_downsample=pipe.temporal_stride, chunk_size=pipe.chunk_size)
        # The I2V prefix is supplied externally, so only future chunks need generation.
        count = math.ceil(5 * pipe.fps / (pipe.temporal_stride * pipe.chunk_size))
        if pipe.input_mode == "i2v":
            # Alaya conditions on pixel 0 externally, then emits 32 pixels per round.
            spans = [(1+i*pipe.chunk_size*pipe.temporal_stride,
                      1+(i+1)*pipe.chunk_size*pipe.temporal_stride) for i in range(count)]
            prompts = [_prompt_for_interval(record['init_decs'], scheduled.events,
                       args.prompt_schedule, start, end, i == 0)
                       for i, (start, end) in enumerate(spans)]
        else:
            spans = [(0 if i == 0 else 1+(i*pipe.chunk_size-1)*pipe.temporal_stride,
                      1+((i+1)*pipe.chunk_size-1)*pipe.temporal_stride) for i in range(count)]
            prompts = scheduled.chunk_prompts[:count]
        if len(prompts) != count:
            raise ValueError("Native prompt count does not match generation chunks")
        image = args.image_dir / f"{record['sample_id']}_seed{seed}.png" if args.image_dir else None
        print(f"[{args.model}] sample={record['sample_id']} seed={seed} prompts={prompts}", flush=True)
        frames = pipe.generate(prompts, seed, image)
        if len(frames) < 5 * pipe.fps + 1:
            raise RuntimeError(f"Generated only {len(frames)} frames; need {5*pipe.fps+1}")
        save_video(frames, pipe.fps, out)
        metadata = {"model": args.model, "checkpoint": str(checkpoint), "dataset": str(args.dataset.resolve()),
                    "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    "sample_id": record['sample_id'], "seed": seed, "schedule": args.prompt_schedule,
                    "input_mode": pipe.input_mode, "initial_image": str(image) if image else None,
                    "native_fps": pipe.fps, "native_temporal_stride": pipe.temporal_stride,
                    "native_chunk_size": pipe.chunk_size, "output_fps": 16, "output_frames": 81,
                    "prompt_policy": "existing schedule_record over native latent chunk spans; first chunk initial-only",
                    "chunk_prompts": prompts,
                    "chunk_seconds": pipe.temporal_stride*pipe.chunk_size/pipe.fps,
                    "chunk_start_seconds": [start/pipe.fps for start, _ in spans],
                    "chunk_end_seconds": [end/pipe.fps for _, end in spans],
                    "conditioning_note": "Interval-overlap schedule at native chunk resolution; not 0.25-second control."}
        if image:
            metadata['initial_image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
            source = image.with_suffix('.json')
            if source.is_file(): metadata['initial_image_provenance'] = json.loads(source.read_text())
        out.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
        print(f"Saved {out}", flush=True)

if __name__ == "__main__":
    main()
