"""Package-native Exp-1 trainer with TensorBoard metrics."""

from __future__ import annotations

import gc
import ctypes
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from causal_prompt.prompt.schedule import (
    CausalPromptTrainingDataset,
    collate_causal_prompt_batch,
    encode_block_prompt_batch,
)

from .model import CausalPromptDMD
from .utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from .checkpoint import (
    latest_checkpoint,
    load_component,
    save_component,
    validate_manifest,
    write_manifest,
)
from .utils.misc import set_seed


def _cycle(loader):
    while True:
        yield from loader


def _release_cpu_memory() -> None:
    """Return freed checkpoint tensors to the Slurm memory cgroup."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _cgroup_memory() -> tuple[float, float]:
    """Return Slurm-cgroup remaining and total memory in GB."""
    cgroup_root = Path("/sys/fs/cgroup")
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            cgroup = cgroup_root / line.removeprefix("0::").lstrip("/")
            while cgroup != cgroup_root.parent:
                current_path = cgroup / "memory.current"
                maximum_path = cgroup / "memory.max"
                if current_path.is_file() and maximum_path.is_file():
                    maximum_text = maximum_path.read_text().strip()
                    if maximum_text != "max":
                        current = int(current_path.read_text().strip())
                        maximum = int(maximum_text)
                        return max(0, maximum - current) / 10**9, maximum / 10**9
                cgroup = cgroup.parent
            break
    values = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    total = values["MemTotal"]
    return values["MemAvailable"] / 10**9, total / 10**9


def _limited_activation_offload(max_gib: float):
    """Offload only a bounded amount of non-leaf activations to CPU."""
    if max_gib <= 0:
        return nullcontext()
    budget = int(max_gib * 1024**3)
    used = 0

    def pack(tensor):
        nonlocal used
        size = tensor.numel() * tensor.element_size()
        if tensor.is_cuda and not tensor.is_leaf and used + size <= budget:
            used += size
            return (tensor.detach().cpu(), tensor.device)
        return (tensor, None)

    def unpack(packed):
        tensor, device = packed
        return tensor if device is None else tensor.to(device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _checkpoint_state(checkpoint_path: str | Path, key: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if key not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} has no {key!r} state dict.")
    return {
        name.replace("model._fsdp_wrapped_module.", "model.", 1)
        if name.startswith("model._fsdp_wrapped_module.")
        else name: value
        for name, value in checkpoint[key].items()
    }


class Exp1Trainer:
    def __init__(self, config, run_dir: Path):
        print("Initializing distributed training...", flush=True)
        launch_distributed_job()
        self.config = config
        self.run_dir = run_dir
        self.rank = dist.get_rank()
        self.is_main = self.rank == 0
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.step = 0
        self.batches_consumed = 0
        self.single_gpu_teacher_staging = dist.get_world_size() == 1
        set_seed(int(config.seed) + self.rank)
        self._status(f"Distributed training initialized ({dist.get_world_size()} GPU).")

        self._status("Loading training models...")
        self.model = CausalPromptDMD(config, self.device)
        self._status("Training models loaded.")
        wrap = dict(
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy="size",
        )
        self._status("Preparing Generator Model for training...")
        self.model.generator = fsdp_wrap(
            self.model.generator,
            **wrap,
            cpu_offload=bool(getattr(config, "generator_cpu_offload", False)),
        )
        # The 14B teacher is frozen and used only by no-grad forwards. Keep it
        # on CPU between calls so a single H100 has room for generator backward.
        if self.single_gpu_teacher_staging:
            # FSDP previously cast this model to BF16 for each forward. The
            # explicitly staged single-GPU teacher must be cast once here.
            self.model.real_score.to(
                device=torch.device("cuda", self.device), dtype=self.dtype
            )
            self._status("Real Score Model prepared for CPU staging.")
        else:
            # The teacher is frozen, so keeping FP32 master shards only wastes
            # VRAM. Cast once on CPU before FSDP moves BF16 shards to each GPU.
            self.model.real_score.to(dtype=self.dtype)
            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                **wrap,
                cpu_offload=bool(config.real_score_cpu_offload),
            )
            self._status("Real Score Model prepared.")
        self.model.fake_score = fsdp_wrap(self.model.fake_score, **wrap)
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            **wrap,
            cpu_offload=bool(config.text_encoder_cpu_offload),
        )
        self._status("All models prepared for training.")
        self._status(f"Loading Generator Checkpoint: {config.generator_ckpt}")
        state = _checkpoint_state(config.generator_ckpt, config.generator_ckpt_key)
        self.model.generator.load_state_dict(state, strict=True)
        del state
        gc.collect()
        self._status("Generator Checkpoint loaded.")

        self._status("Creating optimizers...")
        self.generator_optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.generator.parameters() if parameter.requires_grad],
            lr=float(config.lr),
            betas=(float(config.beta1), float(config.beta2)),
            weight_decay=float(config.weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.fake_score.parameters() if parameter.requires_grad],
            lr=float(config.lr_critic),
            betas=(float(config.beta1_critic), float(config.beta2_critic)),
            weight_decay=float(config.weight_decay),
        )
        self._status("Optimizers ready.")
        self._status(f"Loading Training Dataset: {config.data_path}")
        dataset = CausalPromptTrainingDataset(
            config.data_path,
            config.prompt_schedule,
            fps=float(config.fps),
            temporal_downsample=int(config.temporal_downsample),
            chunk_size=1,
            expected_latent_frames=21,
        )
        self._status(f"Training Dataset loaded: {len(dataset)} samples.")
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        loader = DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            sampler=sampler,
            num_workers=int(config.num_workers),
            collate_fn=collate_causal_prompt_batch,
        )
        self.loader = loader
        self.batches = _cycle(loader)
        self.writer = SummaryWriter(str(run_dir / "tensorboard")) if self.is_main else None
        self.ema = None
        self.max_generator_norm = float(config.max_grad_norm_generator)
        self.max_critic_norm = float(config.max_grad_norm_critic)
        self._resume_if_available()
        self._status(f"TensorBoard Logs: {run_dir / 'tensorboard'}")
        self._status(f"Trainer ready for {config.max_steps} steps.")

    def _status(self, message: str) -> None:
        vram_rest, vram_total = torch.cuda.mem_get_info(self.device)
        ram_rest, ram_total = _cgroup_memory()
        print(
            f"{message} (VRAM {vram_rest / 10**9:.1f}/{vram_total / 10**9:.1f} GB, "
            f"RAM {ram_rest:.1f}/{ram_total:.1f} GB)",
            flush=True,
        )

    def _conditions(self, batch):
        prompts = batch["prompts"]
        with torch.no_grad():
            full = self.model.text_encoder(text_prompts=prompts)
            blocks = encode_block_prompt_batch(
                self.model.text_encoder, batch["block_prompts"]
            )
            negative = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * len(prompts)
            )
        return full, blocks, {key: value.detach() for key, value in negative.items()}

    def _shape(self, batch):
        shape = [int(value) for value in self.config.image_or_video_shape]
        shape[0] = len(batch["prompts"])
        return shape

    def _next_batch(self):
        batch = next(self.batches)
        self.batches_consumed += 1
        return batch

    def _resume_if_available(self) -> None:
        checkpoint_dir = latest_checkpoint(self.run_dir)
        if checkpoint_dir is None:
            print("No training checkpoint found; starting from step 0.", flush=True)
            return
        if (checkpoint_dir / "COMPLETE").is_file():
            self._resume_split(checkpoint_dir)
            return
        checkpoint_path = checkpoint_dir / "model.pt"
        print(f"Resuming Legacy Checkpoint: {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step_from_path = int(checkpoint_dir.name.removeprefix("step_"))
        if int(state.get("format_version", 1)) < 2:
            generator_state = state.get("generator", state.get("generator_ema"))
            if generator_state is None:
                raise KeyError(f"No generator state in {checkpoint_path}")
            self.model.generator.load_state_dict(generator_state, strict=True)
            self.step = step_from_path
            self.batches_consumed = 0
            if self.step >= int(self.config.ema_start_step):
                self.ema = EMA_FSDP(
                    self.model.generator, decay=float(self.config.ema_weight)
                )
            print(
                "Legacy checkpoint restored: Generator weights and step loaded; "
                "critic and optimizers start fresh.",
                flush=True,
            )
            del state
            _release_cpu_memory()
            return

        self.model.generator.load_state_dict(state["generator"], strict=True)
        self.model.fake_score.load_state_dict(state["fake_score"], strict=True)
        self.generator_optimizer.load_state_dict(state["generator_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.step = int(state["step"])
        self.batches_consumed = int(state.get("batches_consumed", 0))
        if state.get("ema") is not None:
            self.ema = EMA_FSDP(
                self.model.generator, decay=float(self.config.ema_weight)
            )
            self.ema.load_state_dict(state["ema"])
        # Restore the deterministic dataset cursor. The sampler ordering is
        # fixed, so only the position within one loader pass is required.
        skip_batches = self.batches_consumed % len(self.loader)
        for _ in range(skip_batches):
            next(self.batches)
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        del state
        _release_cpu_memory()
        print(f"Training state restored at step {self.step}.", flush=True)

    def _resume_split(self, checkpoint_dir: Path) -> None:
        manifest = validate_manifest(checkpoint_dir, verify_hashes=False)
        print(f"Resuming Split Checkpoint: {checkpoint_dir}", flush=True)

        component = load_component(checkpoint_dir, "generator")
        self.model.generator.load_state_dict(component, strict=True)
        del component
        _release_cpu_memory()

        component = load_component(checkpoint_dir, "fake_score")
        self.model.fake_score.load_state_dict(component, strict=True)
        del component
        _release_cpu_memory()

        component = load_component(checkpoint_dir, "generator_optimizer")
        self.generator_optimizer.load_state_dict(component)
        del component
        _release_cpu_memory()

        component = load_component(checkpoint_dir, "critic_optimizer")
        self.critic_optimizer.load_state_dict(component)
        del component
        _release_cpu_memory()

        if "ema.pt" in manifest["files"]:
            component = load_component(checkpoint_dir, "ema")
            self.ema = EMA_FSDP(
                self.model.generator,
                decay=float(self.config.ema_weight),
                initialize=False,
            )
            self.ema.load_state_dict(component)
            del component
            _release_cpu_memory()

        rng = load_component(checkpoint_dir, "rng_state")
        self.step = int(manifest["step"])
        self.batches_consumed = int(manifest.get("batches_consumed", 0))
        skip_batches = self.batches_consumed % len(self.loader)
        for _ in range(skip_batches):
            next(self.batches)
        if rng.get("python_rng_state") is not None:
            random.setstate(rng["python_rng_state"])
        if rng.get("numpy_rng_state") is not None:
            np.random.set_state(rng["numpy_rng_state"])
        if rng.get("torch_rng_state") is not None:
            torch.set_rng_state(rng["torch_rng_state"])
        if rng.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(rng["cuda_rng_state_all"])
        del rng
        _release_cpu_memory()
        print(f"Training state restored at step {self.step}.", flush=True)

    def _generator_step(self, batch):
        self._status(f"Step {self.step + 1}: Generator update started.")
        if self.single_gpu_teacher_staging:
            self._status("Moving Real Score Model to GPU...")
            self.model.real_score.to(
                device=torch.device("cuda", self.device), dtype=self.dtype
            )
            self._status("Real Score Model is on GPU.")
        full, blocks, negative = self._conditions(batch)
        self.generator_optimizer.zero_grad(set_to_none=True)
        offload_gib = float(self.config.activation_cpu_offload_gib)
        activation_context = _limited_activation_offload(offload_gib)
        if offload_gib > 0:
            print(
                f"Step {self.step + 1}: CPU-offloading at most "
                f"{offload_gib:g} GiB of backward activations.",
                flush=True,
            )
        with activation_context:
            loss, metrics = self.model.generator_loss(
                self._shape(batch), full, negative, blocks
            )
        if self.single_gpu_teacher_staging:
            self._status("Moving Real Score Model back to CPU before backward...")
            self.model.real_score.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self._status("Real Score Model is on CPU; starting backward.")
        # The frozen real/fake score forwards have finished. Return their
        # inactive allocator blocks before the generator's activation-heavy
        # backward pass.
        torch.cuda.empty_cache()
        loss.backward()
        norm = self.model.generator.clip_grad_norm_(self.max_generator_norm)
        self.generator_optimizer.step()
        self._status(f"Step {self.step + 1}: Generator update finished.")
        if self.ema is not None:
            self.ema.update(self.model.generator)
        return {
            "generator_loss": loss.detach(),
            "generator_grad_norm": norm.detach(),
            **metrics,
        }

    def _critic_step(self, batch):
        self._status(f"Step {self.step + 1}: Critic update started.")
        full, blocks, _ = self._conditions(batch)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.model.critic_loss(self._shape(batch), full, blocks)
        loss.backward()
        norm = self.model.fake_score.clip_grad_norm_(self.max_critic_norm)
        self.critic_optimizer.step()
        self._status(f"Step {self.step + 1}: Critic update finished.")
        return {
            "critic_loss": loss.detach(),
            "critic_grad_norm": norm.detach(),
            **metrics,
        }

    def _save(self):
        self._status(f"Saving checkpoint at step {self.step}...")
        generator = fsdp_state_dict(self.model.generator)
        checkpoint_dir = self.run_dir / "checkpoints" / f"step_{self.step:06d}"
        if self.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_component(checkpoint_dir, "generator", generator)
        del generator
        _release_cpu_memory()

        fake_score = fsdp_state_dict(self.model.fake_score)
        if self.is_main:
            save_component(checkpoint_dir, "fake_score", fake_score)
            save_component(
                checkpoint_dir,
                "generator_optimizer",
                self.generator_optimizer.state_dict(),
            )
            save_component(
                checkpoint_dir,
                "critic_optimizer",
                self.critic_optimizer.state_dict(),
            )
            names = [
                "generator",
                "fake_score",
                "generator_optimizer",
                "critic_optimizer",
            ]
            if self.ema is not None:
                save_component(checkpoint_dir, "ema", self.ema.state_dict())
                names.append("ema")
            rng = {
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            }
            save_component(checkpoint_dir, "rng_state", rng)
            names.append("rng_state")
            write_manifest(
                checkpoint_dir,
                {"step": self.step, "batches_consumed": self.batches_consumed},
                names,
            )
            self._status(f"Checkpoint saved: {checkpoint_dir}")
        del fake_score
        _release_cpu_memory()

    def _log(self, metrics, elapsed):
        if not self.is_main:
            return
        scalars = {name: float(value) for name, value in metrics.items()}
        scalars["iteration_seconds"] = elapsed
        if self.step % int(self.config.log_steps) == 0:
            formatted = " ".join(
                f"{name}={value:.6g}" for name, value in sorted(scalars.items())
            )
            print(f"train step={self.step} {formatted}", flush=True)
        if self.writer is None:
            return
        for name, value in metrics.items():
            self.writer.add_scalar(f"train/{name}", float(value), self.step)
        self.writer.add_scalar("system/iteration_seconds", elapsed, self.step)
        self.writer.add_scalar("train/generator_lr", float(self.config.lr), self.step)
        self.writer.add_scalar("train/critic_lr", float(self.config.lr_critic), self.step)
        if self.step % int(self.config.tensorboard_flush_steps) == 0:
            self.writer.flush()

    def train(self):
        self.model.eval()
        self._status(f"Training started: {self.config.max_steps} total steps.")
        while self.step < int(self.config.max_steps):
            started = time.monotonic()
            metrics = {}
            if self.step % int(self.config.dfake_gen_update_ratio) == 0:
                metrics.update(self._generator_step(self._next_batch()))
            metrics.update(self._critic_step(self._next_batch()))
            self.step += 1
            if self.step >= int(self.config.ema_start_step) and self.ema is None:
                self.ema = EMA_FSDP(
                    self.model.generator, decay=float(self.config.ema_weight)
                )
            self._log(metrics, time.monotonic() - started)
            if self.step % int(self.config.checkpoint_steps) == 0:
                self._save()
            if self.step % int(self.config.gc_interval) == 0:
                gc.collect()
                torch.cuda.empty_cache()
        self._save()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self._status(f"Training completed at step {self.step}.")
