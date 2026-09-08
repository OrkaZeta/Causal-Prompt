"""Frame-wise Causal Forcing DMD model for Exp 1."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..pipeline.self_forcing_training import SelfForcingTrainingPipeline
from ..utils.loss import get_denoising_loss
from ..utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder




class CausalPromptDMD(nn.Module):
    """The 21-frame CF DMD path with scheduled generator conditioning."""

    def __init__(self, config, device: torch.device | int):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.num_frame_per_block = int(config.num_frame_per_block)
        self.num_training_frames = int(config.num_training_frames)
        if self.num_training_frames != 21 or self.num_frame_per_block != 1:
            raise ValueError("Exp 1 requires exactly 21 frame-wise latent blocks.")

        model_kwargs = dict(getattr(config, "model_kwargs", {}))
        model_kwargs["model_path"] = str(config.base_model_path)
        print(f"Loading Generator Base Model: {config.base_model_path}", flush=True)
        self.generator = WanDiffusionWrapper(is_causal=True, **model_kwargs)
        self.generator.model.requires_grad_(True)
        print("Generator Base Model loaded.", flush=True)
        print(f"Loading Real Score Model: {config.real_model_path}", flush=True)
        self.real_score = WanDiffusionWrapper(
            model_name=str(config.real_name),
            model_path=str(config.real_model_path),
            is_causal=False,
            timestep_shift=float(config.timestep_shift),
        )
        self.real_score.model.requires_grad_(False)
        print("Real Score Model loaded.", flush=True)
        print(f"Loading Fake Score Model: {config.base_model_path}", flush=True)
        self.fake_score = WanDiffusionWrapper(
            model_path=str(config.base_model_path),
            is_causal=False,
            timestep_shift=float(config.timestep_shift),
        )
        self.fake_score.model.requires_grad_(True)
        print("Fake Score Model loaded.", flush=True)
        print(f"Loading Text Encoder: {config.base_model_path}", flush=True)
        self.text_encoder = WanTextEncoder(str(config.base_model_path))
        self.text_encoder.requires_grad_(False)
        print("Text Encoder loaded.", flush=True)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        denoising_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
        if config.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0.0]))
            )
            denoising_steps = timesteps[1000 - denoising_steps]
        self.denoising_step_list = denoising_steps.to(device)

        if config.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled.", flush=True)
        self.inference_pipeline: Optional[SelfForcingTrainingPipeline] = None
        self.denoising_loss_func = get_denoising_loss(config.denoising_loss_type)()
        self.num_train_timestep = int(config.num_train_timestep)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = float(config.guidance_scale)
        self.fake_guidance_scale = float(getattr(config, "fake_guidance_scale", 0.0))
        self.timestep_shift = float(config.timestep_shift)
        self.ts_schedule = bool(getattr(config, "ts_schedule", True))
        self.ts_schedule_max = bool(getattr(config, "ts_schedule_max", False))
        self.min_score_timestep = int(getattr(config, "min_score_timestep", 0))
        alphas = getattr(self.scheduler, "alphas_cumprod", None)
        self.scheduler.alphas_cumprod = None if alphas is None else alphas.to(device)
        print("DMD models are ready.", flush=True)

    def _score_timestep(self, batch_size: int, frame_count: int, low: int, high: int):
        timestep = torch.randint(
            low,
            high,
            (batch_size, 1),
            device=self.device,
            dtype=torch.long,
        ).repeat(1, frame_count)
        if self.timestep_shift > 1:
            scaled = timestep / 1000
            timestep = (
                self.timestep_shift
                * scaled
                / (1 + (self.timestep_shift - 1) * scaled)
                * 1000
            )
        return timestep.clamp(self.min_step, self.max_step)

    def _pipeline(self) -> SelfForcingTrainingPipeline:
        if self.inference_pipeline is None:
            self.inference_pipeline = SelfForcingTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=1,
                independent_first_frame=False,
                same_step_across_blocks=True,
                last_step_only=False,
                num_max_frames=21,
                context_noise=int(self.config.context_noise),
            )
        return self.inference_pipeline

    def _run_generator(self, shape, conditional_dict, block_conditional_dicts):
        noise = torch.randn(shape, device=self.device, dtype=self.dtype)
        generated, step_from, step_to = self._pipeline().inference_with_trajectory(
            noise=noise,
            block_conditional_dicts=block_conditional_dicts,
            **conditional_dict,
        )
        return generated.to(self.dtype), step_from, step_to

    def _kl_gradient(self, noisy, estimated_clean, timestep, conditional, unconditional):
        _, fake_cond = self.fake_score(
            noisy_image_or_video=noisy,
            conditional_dict=conditional,
            timestep=timestep,
        )
        if self.fake_guidance_scale:
            _, fake_uncond = self.fake_score(
                noisy_image_or_video=noisy,
                conditional_dict=unconditional,
                timestep=timestep,
            )
            fake = fake_cond + (fake_cond - fake_uncond) * self.fake_guidance_scale
        else:
            fake = fake_cond
        _, real_cond = self.real_score(
            noisy_image_or_video=noisy,
            conditional_dict=conditional,
            timestep=timestep,
        )
        _, real_uncond = self.real_score(
            noisy_image_or_video=noisy,
            conditional_dict=unconditional,
            timestep=timestep,
        )
        real = real_cond + (real_cond - real_uncond) * self.real_guidance_scale
        gradient = fake - real
        normalizer = (estimated_clean - real).abs().mean(
            dim=(1, 2, 3, 4), keepdim=True
        )
        gradient = torch.nan_to_num(gradient / normalizer)
        return gradient, {
            "dmd_gradient_norm": gradient.abs().mean().detach(),
            "score_timestep": timestep.detach().float().mean(),
        }

    def generator_loss(self, shape, conditional, unconditional, block_conditionals):
        generated, step_from, step_to = self._run_generator(
            shape, conditional, block_conditionals
        )
        low = step_to if self.ts_schedule and step_to is not None else self.min_score_timestep
        high = step_from if self.ts_schedule_max and step_from is not None else self.num_train_timestep
        with torch.no_grad():
            timestep = self._score_timestep(shape[0], shape[1], low, high)
            noise = torch.randn_like(generated)
            noisy = self.scheduler.add_noise(
                generated.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
            ).detach().unflatten(0, generated.shape[:2])
            gradient, metrics = self._kl_gradient(
                noisy, generated, timestep, conditional, unconditional
            )
        detached_gradient = gradient.detach().to(generated.dtype)
        surrogate = (generated * detached_gradient).mean()
        reported = 0.5 * detached_gradient.float().square().mean()
        loss = surrogate + reported - surrogate.detach()
        return loss, metrics

    def critic_loss(self, shape, conditional, block_conditionals):
        with torch.no_grad():
            generated, step_from, step_to = self._run_generator(
                shape, conditional, block_conditionals
            )
        low = step_to if self.ts_schedule and step_to is not None else self.min_score_timestep
        high = step_from if self.ts_schedule_max and step_from is not None else self.num_train_timestep
        timestep = self._score_timestep(shape[0], shape[1], low, high)
        noise = torch.randn_like(generated)
        noisy = self.scheduler.add_noise(
            generated.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, generated.shape[:2])
        _, prediction = self.fake_score(
            noisy_image_or_video=noisy,
            conditional_dict=conditional,
            timestep=timestep,
        )
        flow = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=prediction.flatten(0, 1),
            xt=noisy.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        )
        loss = self.denoising_loss_func(
            x=generated.flatten(0, 1),
            x_pred=prediction.flatten(0, 1),
            noise=noise.flatten(0, 1),
            noise_pred=None,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=timestep.flatten(0, 1),
            flow_pred=flow,
        )
        return loss, {"critic_timestep": timestep.detach().float().mean()}
