# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import binascii
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ["cache_video"]


def _random_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        name += suffix if suffix.startswith(".") else f".{suffix}"
    return name


def cache_video(
    tensor,
    save_file=None,
    fps=16,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5,
):
    cache_file = (
        osp.join("/tmp", _random_name(suffix=suffix))
        if save_file is None
        else save_file
    )
    error = None
    for _ in range(retry):
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
            error = exception
    print(f"cache_video failed, error: {error}", flush=True)
    return None
