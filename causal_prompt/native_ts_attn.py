"""TS-Attn-inspired temporal routing for the native Wan FlashAttention runtime.

The reference TS-Attn implementation materializes attention maps in Diffusers.
This adapter preserves its central temporal event-routing idea while keeping the
native LightX2V Wan runtime and FlashAttention kernels.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch


MAX_EVENTS = 3


def _find_token_span(prompt_ids: torch.Tensor, token_ids: list[int]) -> list[int]:
    if not token_ids:
        return []
    needle = torch.tensor(token_ids, dtype=prompt_ids.dtype)
    for start in range(prompt_ids.numel() - needle.numel() + 1):
        if torch.equal(prompt_ids[start : start + needle.numel()].cpu(), needle):
            return list(range(start, start + needle.numel()))
    return []


def _native_temporal_cross_attention(self, x, context, context_lens):
    """WanCrossAttention.forward with fixed-shape temporal event routing."""
    from wan.modules.attention import flash_attention

    batch, heads, head_dim = x.size(0), self.num_heads, self.head_dim
    query = self.norm_q(self.q(x)).view(batch, -1, heads, head_dim)
    key = self.norm_k(self.k(context)).view(batch, -1, heads, head_dim)
    value = self.v(context).view(batch, -1, heads, head_dim)

    if self._ts_active:
        output = torch.zeros_like(query)
        for route in range(self._ts_event_count):
            routed_value = value * self._ts_value_masks[route].view(1, -1, 1, 1)
            routed = flash_attention(
                query,
                key,
                routed_value,
                k_lens=context_lens,
            )
            output.add_(
                routed * self._ts_query_masks[route].view(1, -1, 1, 1)
            )
    else:
        output = flash_attention(query, key, value, k_lens=context_lens)

    return self.o(output.flatten(2))


@dataclass
class NativeTemporalRouting:
    """Install and configure temporal event routing on a native ``WanT2V``."""

    pipeline: Any
    frame_num: int = 81
    width: int = 832
    height: int = 480
    control_steps: int = 1
    event_scale: float = 1.25

    def __post_init__(self) -> None:
        if not 1 <= self.control_steps <= 4:
            raise ValueError("control_steps must be in [1, 4]")
        self._modules = []
        self._step_index = -1
        self._has_temporal_events = False
        self._install_attention_routes()
        self._install_step_hook()

    @property
    def sequence_length(self) -> int:
        latent_frames = (self.frame_num - 1) // self.pipeline.vae_stride[0] + 1
        latent_height = self.height // self.pipeline.vae_stride[1]
        latent_width = self.width // self.pipeline.vae_stride[2]
        patch_height, patch_width = self.pipeline.patch_size[1:]
        return latent_frames * (latent_height // patch_height) * (
            latent_width // patch_width
        )

    @property
    def latent_frames(self) -> int:
        return (self.frame_num - 1) // self.pipeline.vae_stride[0] + 1

    def _install_attention_routes(self) -> None:
        sequence_length = self.sequence_length
        for model in (self.pipeline.high_noise_model, self.pipeline.low_noise_model):
            for index, block in enumerate(model.blocks):
                if index % 2 == 0:
                    continue
                module = block.cross_attn
                module.register_buffer(
                    "_ts_value_masks",
                    torch.ones(
                        MAX_EVENTS,
                        self.pipeline.config.text_len,
                        device=module.q.weight.device,
                        dtype=module.q.weight.dtype,
                    ),
                    persistent=False,
                )
                module.register_buffer(
                    "_ts_query_masks",
                    torch.zeros(
                        MAX_EVENTS,
                        sequence_length,
                        device=module.q.weight.device,
                        dtype=module.q.weight.dtype,
                    ),
                    persistent=False,
                )
                module._ts_active = False
                module._ts_event_count = 1
                module.forward = types.MethodType(
                    _native_temporal_cross_attention, module
                )
                self._modules.append(module)

    def _install_step_hook(self) -> None:
        original = self.pipeline._prepare_model_for_timestep
        controller = self

        def prepare_model(pipeline_self, timestep, boundary, offload_model):
            controller._step_index += 1
            controller._set_active(
                controller._has_temporal_events
                and controller._step_index < controller.control_steps
            )
            return original(timestep, boundary, offload_model)

        self.pipeline._prepare_model_for_timestep = types.MethodType(
            prepare_model, self.pipeline
        )

    def _set_active(self, active: bool) -> None:
        for module in self._modules:
            module._ts_active = active

    def configure(self, prompt: str, events: tuple[str, ...]) -> None:
        """Configure fixed-shape token and video masks for one prompt."""
        if not 1 <= len(events) <= MAX_EVENTS:
            raise ValueError(f"Expected 1-{MAX_EVENTS} events, got {len(events)}")

        tokenizer = self.pipeline.text_encoder.tokenizer.tokenizer
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=self.pipeline.config.text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        event_spans = []
        for event in events:
            event_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(event))
            span = _find_token_span(prompt_ids, event_ids)
            if not span:
                raise ValueError(
                    f"TS-Attn event tokens were not found in the prompt: {event!r}"
                )
            event_spans.append(span)

        value_masks = torch.ones(MAX_EVENTS, self.pipeline.config.text_len)
        for route, current_span in enumerate(event_spans):
            for other_route, other_span in enumerate(event_spans):
                value_masks[route, other_span] = (
                    self.event_scale if route == other_route else 0.0
                )

        query_masks = torch.zeros(MAX_EVENTS, self.sequence_length)
        frames_per_event = [self.latent_frames // len(events)] * len(events)
        for index in range(self.latent_frames % len(events)):
            frames_per_event[index] += 1
        tokens_per_frame = self.sequence_length // self.latent_frames
        start_frame = 0
        for route, frame_count in enumerate(frames_per_event):
            start = start_frame * tokens_per_frame
            end = (start_frame + frame_count) * tokens_per_frame
            query_masks[route, start:end] = 1.0
            start_frame += frame_count
        if not torch.allclose(query_masks.sum(dim=0), torch.ones(self.sequence_length)):
            raise RuntimeError("Temporal routing masks do not cover the video sequence")

        for module in self._modules:
            module._ts_event_count = len(events)
            module._ts_value_masks.copy_(
                value_masks.to(module._ts_value_masks.device)
            )
            module._ts_query_masks.copy_(
                query_masks.to(module._ts_query_masks.device)
            )
        self._step_index = -1
        # A single event has no temporal leakage to suppress.
        self._has_temporal_events = len(events) > 1
        self._set_active(self._has_temporal_events and self.control_steps > 0)


def control_steps_from_ratio(ratio: float, sampling_steps: int = 4) -> int:
    """Convert a user ratio to at least one controlled distilled step."""
    return max(1, min(sampling_steps, math.ceil(ratio * sampling_steps)))
