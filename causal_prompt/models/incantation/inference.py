"""Project-owned adapter around the unmodified Incantation preview runtime."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


class IncantationPipeline:
    fps = 16
    temporal_stride = 4
    chunk_size = 1
    height = 256
    width = 448
    input_mode = "i2v"

    def __init__(self, checkpoint: Path, base_model: Path):
        source = Path(__file__).resolve().parents[3] / "third_party" / "Incantation"
        if not source.is_dir():
            raise FileNotFoundError(f"Incantation source does not exist: {source}")
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location("_incantation_inference", source / "inference.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Incantation inference module from {source}")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        self.upstream = upstream

        from modules.causal_model import CausalWanModel
        from safetensors.torch import load_file
        from utils.scheduler import FlowMatchScheduler
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE

        required = {
            "checkpoint": checkpoint,
            "T5 checkpoint": base_model / "models_t5_umt5-xxl-enc-bf16.pth",
            "T5 tokenizer": base_model / "google" / "umt5-xxl",
            "VAE checkpoint": base_model / "Wan2.2_VAE.pth",
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"Incantation {label} does not exist: {path}")

        device, dtype = torch.device("cuda"), torch.bfloat16
        self.text_encoder = T5EncoderModel(text_len=512, dtype=dtype, device=torch.device("cpu"),
            checkpoint_path=str(required["T5 checkpoint"]), tokenizer_path=str(required["T5 tokenizer"]))
        self.vae = Wan2_2_VAE(vae_pth=str(required["VAE checkpoint"]), device=device)
        self.vae.model.cpu()
        model = CausalWanModel(model_type="ti2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=48, dim=3072, ffn_dim=14336, freq_dim=256, text_dim=4096,
            out_dim=48, num_heads=24, num_layers=30, cross_attn_norm=True, eps=1e-6)
        state = load_file(str(checkpoint), device="cpu")
        cleaned = {}
        for key, value in state.items():
            key = key.replace("._mod.", ".")
            cleaned[key[6:] if key.startswith("model.") else key] = value
        model.load_state_dict(cleaned, strict=True)
        print(f"[WM-Incantation] {checkpoint}: strict loaded={len(cleaned)} missing=0 unexpected=0", flush=True)
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def generate(self, prompts, seed, image=None):
        if image is None:
            raise ValueError("Incantation requires an initial-state image")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.cpu()
        self.text_encoder.model.cuda()
        contexts, count = self.upstream.encode_prompts(prompts, self.text_encoder, self.device, len(prompts))
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        self.vae.model.cuda()
        pixels = self.upstream.load_image(str(image), self.height, self.width).to(self.device)
        initial = self.vae.encode([pixels.float()])[0].unsqueeze(0).to(dtype=self.dtype)
        self.vae.model.cpu()
        self.model.cuda()
        torch.cuda.empty_cache()
        latent = self.upstream.generate(self.model, self.scheduler, initial, contexts, count,
            [1000, 750, 500, 250], self.device, self.dtype, kv_window_size=12)
        self.model.cpu()
        self.vae.model.cuda()
        video = self.vae.decode([latent.squeeze(0).float()])[0]
        frames = ((video.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu()
        self.vae.model.cpu()
        self.model.cuda()
        torch.cuda.empty_cache()
        return frames
