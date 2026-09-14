"""Generate AlayaWorld conditioning images from initial descriptions only."""
import argparse
import json
from pathlib import Path


def generate_initial_images(records, seeds, save_dir, checkpoint):
    """Create missing initial images for the exact records/seeds being generated."""
    jobs = [(record, seed, save_dir / f"{record['sample_id']}_seed{seed}.png")
            for record in records for seed in seeds]
    jobs = [(record, seed, out) for record, seed, out in jobs
            if not (out.is_file() and out.with_suffix(".json").is_file())]
    if not jobs:
        return

    import torch
    from PIL import Image
    from causal_prompt.models.wan22 import WanT2V5B
    from causal_prompt.models.wan22.configs import WAN_CONFIGS

    save_dir.mkdir(parents=True, exist_ok=True)
    pipe = WanT2V5B(config=WAN_CONFIGS["t2v-5B"], checkpoint_dir=str(checkpoint),
                    device_id=0, rank=0, t5_cpu=True, convert_model_dtype=True)
    for record, seed, out in jobs:
        # Deliberately do not pass events or the full video caption into this model.
        with torch.inference_mode():
            video = pipe.generate(record["init_decs"], size=(1280, 704), frame_num=1,
                    shift=5.0, sample_solver="unipc", sampling_steps=50, guide_scale=5.0,
                    seed=seed, offload_model=True)
        frame = ((video[:, 0].float().clamp(-1, 1) + 1) * 127.5).byte()
        frame = frame.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(frame).save(out)
        out.with_suffix(".json").write_text(json.dumps({"model": "Wan2.2-TI2V-5B",
            "checkpoint": str(checkpoint), "prompt": record["init_decs"], "seed": seed,
            "future_events_used": False, "frame_num": 1}, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved initial-only image {out}", flush=True)
    del pipe
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("/projects/hi-paris/ZiyiData/Models/Wan2.2-TI2V-5B"))
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--split", default="train", choices=["train", "test", "all"])
    args = p.parse_args()
    records = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    records = [r for r in records if args.split == "all" or r['split'] == args.split]
    generate_initial_images(records, args.seed, args.save_dir, args.checkpoint)

if __name__ == '__main__': main()
