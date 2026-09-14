"""Generate AlayaWorld conditioning images from initial descriptions only."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from causal_prompt.models.wan22 import WanT2V5B
from causal_prompt.models.wan22.configs import WAN_CONFIGS


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
    args.save_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, s, args.save_dir/f"{r['sample_id']}_seed{s}.png") for r in records for s in args.seed]
    jobs = [(r,s,out) for r,s,out in jobs if not (out.is_file() and out.with_suffix('.json').is_file())]
    if not jobs: return
    pipe = WanT2V5B(config=WAN_CONFIGS['t2v-5B'], checkpoint_dir=str(args.checkpoint),
                    device_id=0, rank=0, t5_cpu=True, convert_model_dtype=True)
    for record, seed, out in jobs:
        # Deliberately do not pass events or the full video caption into this model.
        with torch.inference_mode():
            video = pipe.generate(record['init_decs'], size=(1280,704), frame_num=1,
                    shift=5.0, sample_solver='unipc', sampling_steps=50, guide_scale=5.0,
                    seed=seed, offload_model=True)
        frame = ((video[:,0].float().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
        Image.fromarray(frame).save(out)
        out.with_suffix('.json').write_text(json.dumps({'model':'Wan2.2-TI2V-5B',
            'checkpoint':str(args.checkpoint), 'prompt':record['init_decs'], 'seed':seed,
            'future_events_used':False,'frame_num':1},ensure_ascii=False,indent=2)+'\n')
        print(f'Saved initial-only image {out}',flush=True)

if __name__ == '__main__': main()
