# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import binascii
import logging
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ["masks_like", "save_video"]


def _random_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        name += suffix if suffix.startswith(".") else f".{suffix}"
    return name


def save_video(
    tensor,
    save_file=None,
    fps=16,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
):
    cache_file = (
        osp.join("/tmp", _random_name(suffix=suffix))
        if save_file is None
        else save_file
    )
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack(
            [
                torchvision.utils.make_grid(
                    frame,
                    nrow=nrow,
                    normalize=normalize,
                    value_range=value_range,
                )
                for frame in tensor.unbind(2)
            ],
            dim=1,
        ).permute(1, 2, 3, 0)
        tensor = (tensor * 255).to(torch.uint8).cpu()
        writer = imageio.get_writer(
            cache_file, fps=fps, codec="libx264", quality=8
        )
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file
    except Exception as exception:
        logging.error("save_video failed: %s", exception)
        return None


def masks_like(tensor, zero=False, generator=None, p=0.2):
    """Build the timestep masks required by the released 5B architecture."""
    if not isinstance(tensor, list):
        raise TypeError("tensor must be a list")
    out1 = [torch.ones_like(value) for value in tensor]
    out2 = [torch.ones_like(value) for value in tensor]
    if zero:
        for first, second in zip(out1, out2):
            if generator is not None:
                random_number = torch.rand(
                    1, generator=generator, device=generator.device
                ).item()
                if random_number < p:
                    first[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=first.device,
                        generator=generator,
                    ).expand_as(first[:, 0]).exp()
                    second[:, 0] = torch.zeros_like(second[:, 0])
            else:
                first[:, 0] = torch.zeros_like(first[:, 0])
                second[:, 0] = torch.zeros_like(second[:, 0])
    return out1, out2
