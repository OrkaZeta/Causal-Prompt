"""Distributed package entry point for Exp-1 Causal Prompt DMD training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "exp1_cp_dmd_framewise.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--schedule", choices=("current", "causal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Training Configuration: {args.config}", flush=True)
    print(f"Training Output: {args.run_dir}", flush=True)
    from .training import Exp1Trainer

    config = OmegaConf.load(args.config)
    activation_offload = os.environ.get("ACTIVATION_CPU_OFFLOAD_GIB")
    if activation_offload is not None:
        config.activation_cpu_offload_gib = float(activation_offload)
        if config.activation_cpu_offload_gib < 0:
            raise ValueError("ACTIVATION_CPU_OFFLOAD_GIB must be non-negative.")
    generator_cpu_offload = os.environ.get("GENERATOR_CPU_OFFLOAD")
    if generator_cpu_offload is not None:
        normalized = generator_cpu_offload.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("GENERATOR_CPU_OFFLOAD must be a boolean value.")
        config.generator_cpu_offload = normalized in {"1", "true", "yes", "on"}


    config.data_path = str(args.data_path.resolve())
    config.prompt_schedule = args.schedule
    config.generator_ckpt = str(args.checkpoint.resolve())
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive.")
    if config.score_prompt_policy != "full":
        raise ValueError("Bidirectional DMD score models require a full prompt.")

    run_dir = args.run_dir.resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, run_dir / "resolved_config.yaml")
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "experiment_name": run_dir.parents[1].name,
                    "experiment_id": run_dir.parent.name,
                    "schedule": args.schedule,
                    "metrics": "tensorboard",
                    "tensorboard_dir": str(run_dir / "tensorboard"),
                    "checkpoint_dir": str(run_dir / "checkpoints"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    trainer = Exp1Trainer(config, run_dir)
    trainer.train()


if __name__ == "__main__":
    main()
