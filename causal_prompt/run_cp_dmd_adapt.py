"""Preflight or launch Exp-1 causal-prompt DMD adaptation."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from causal_prompt.prompt.schedule import CausalPromptTrainingDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "causal_prompt"
    / "models"
    / "causal_forcing"
    / "configs"
    / "exp1_cp_dmd_framewise.yaml"
)


def build_run_dir(
    output_root: Path, experiment_name: str, experiment_id: str, schedule: str
) -> Path:
    """Return outputs/<experiment>/<id>/<schedule> without derived hashes."""
    return (output_root / experiment_name / experiment_id / schedule).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=("current", "causal"), required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "activitynet_causal_5s_train.jsonl",
    )
    parser.add_argument("--sample-id", help="Train on exactly this sample from --dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--exp-name", default="exp1_cp_dmd")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/projects/hi-paris/ZiyiData/Models/Causal-Forcing/framewise/"
            "causal_forcing.pt"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> dict[str, int]:
    dataset = CausalPromptTrainingDataset(
        args.dataset,
        args.schedule,
        fps=16.0,
        temporal_downsample=4,
        chunk_size=1,
        expected_latent_frames=21,
        sample_id=args.sample_id,
    )
    stats: Counter[str] = Counter(samples=len(dataset))
    for index in range(len(dataset)):
        item = dataset[index]
        stats["blocks"] += len(item["block_prompts"])
        stats["unique_block_prompts"] += len(set(item["block_prompts"]))
    return dict(stats)


def main() -> None:
    args = _parse_args()
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive.")
    for label, value in (("exp-name", args.exp_name), ("exp-id", args.exp_id)):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
            raise ValueError(f"--{label} must be a filename-safe identifier.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    stats = _preflight(args)
    print(
        f"Exp-1 preflight passed: schedule={args.schedule}, "
        f"samples={stats['samples']}, blocks={stats['blocks']}, "
        f"unique_sample_prompt_states={stats['unique_block_prompts']}",
        flush=True,
    )
    if args.dry_run:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"4-step CF checkpoint does not exist: {args.checkpoint}"
        )

    output_dir = build_run_dir(
        args.output_root, args.exp_name, args.exp_id, args.schedule
    )
    if output_dir.is_dir() and any(output_dir.iterdir()):
        from causal_prompt.models.causal_forcing.checkpoint import latest_checkpoint

        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileExistsError(
                f"Run directory is non-empty but has no resumable checkpoint: "
                f"{output_dir}"
            )
        print(f"Automatic Resume Checkpoint: {checkpoint}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "causal_prompt.models.causal_forcing.train",
        "--config",
        str(args.config.resolve()),
        "--data-path",
        str(args.dataset.resolve()),
        "--schedule",
        args.schedule,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--run-dir",
        str(output_dir),
    ]
    if args.sample_id is not None:
        command.extend(("--sample-id", args.sample_id))
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
    print(
        f"Exp-1 launch: output_dir={output_dir} nproc={args.nproc_per_node} "
        f"max_steps={args.max_steps or 'config'}",
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
