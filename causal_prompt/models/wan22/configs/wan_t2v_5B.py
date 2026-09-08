# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg


t2v_5B = EasyDict(__name__="Config: Wan 2.2 T2V 5B")
t2v_5B.update(wan_shared_cfg)

t2v_5B.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
t2v_5B.t5_tokenizer = "google/umt5-xxl"
t2v_5B.vae_checkpoint = "Wan2.2_VAE.pth"
t2v_5B.vae_stride = (4, 16, 16)

# The released 5B checkpoint retains its TI2V architecture, but this project
# exposes only its text-only inference path.
t2v_5B.patch_size = (1, 2, 2)
t2v_5B.dim = 3072
t2v_5B.ffn_dim = 14336
t2v_5B.freq_dim = 256
t2v_5B.num_heads = 24
t2v_5B.num_layers = 30
t2v_5B.window_size = (-1, -1)
t2v_5B.qk_norm = True
t2v_5B.cross_attn_norm = True
t2v_5B.eps = 1e-6

t2v_5B.sample_fps = 16
t2v_5B.sample_shift = 5.0
t2v_5B.sample_steps = 50
t2v_5B.sample_guide_scale = 5.0
t2v_5B.frame_num = 81
