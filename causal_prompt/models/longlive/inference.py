"""LongLive 2.0 BF16 inference, backed by the project-owned runtime."""
from pathlib import Path
import os
import sys
import torch


class LongLivePipeline:
    fps = 24
    temporal_stride = 4
    chunk_size = 8
    height = 704
    width = 1280
    input_mode = "t2v"

    def __init__(self, checkpoint: Path, base_model: Path):
        vendor = Path(__file__).with_name("vendor")
        sys.path.insert(0, str(vendor))
        os.environ["LONGLIVE_BASE_MODEL_PATH"] = str(base_model.resolve())
        from omegaconf import OmegaConf
        from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
        from utils.nvfp4_checkpoint import unwrap_generator_state_dict
        self.config = OmegaConf.create({
            "model_kwargs": {"model_name": "Wan2.2-TI2V-5B", "timestep_shift": 5.0,
                             "num_frame_per_block": 8, "local_attn_size": 32},
            "num_frame_per_block": 8, "image_or_video_shape": [1, 32, 48, 44, 80],
            "sampling_steps": 4, "guidance_scale": 1.0, "sink_size": 8,
            "multi_shot_sink": False, "multi_shot_rope_offset": 0.0,
            "streaming_vae": False, "async_vae": False, "vae_type": "wan",
        })
        self.pipe = CausalDiffusionInferencePipeline(self.config, device=torch.device("cuda"))
        path = checkpoint / "model_bf16.pt" if checkpoint.is_dir() else checkpoint
        state = unwrap_generator_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        cleaned = {}
        for key, tensor in state.items():
            name = ".".join(p for p in key.split(".") if p not in {"_fsdp_wrapped_module", "_orig_mod"})
            if name.startswith("module."):
                name = name[7:]
            if name in cleaned:
                raise ValueError(f"Checkpoint key collision: {name}")
            cleaned[name] = tensor
        expected = self.pipe.generator.state_dict()
        # Releases may store either the wrapper or its underlying DiT.
        if set(cleaned) != set(expected) and {"model." + k for k in cleaned} == set(expected):
            cleaned = {"model." + k: v for k, v in cleaned.items()}
        self.pipe.generator.load_state_dict(cleaned, strict=True)
        print(f"[WM-LongLive] {path}: strict loaded={len(cleaned)} missing=0 unexpected=0", flush=True)
        del state, cleaned, expected
        self.pipe.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        # Text and VAE are staged separately from the DiT to bound peak VRAM.
        self.pipe.text_encoder.cpu()
        self.pipe.vae.cpu()
        self.pipe.generator.cuda()

    @torch.inference_mode()
    def generate(self, prompts, seed, image=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.pipe.generator.cpu()
        self.pipe.text_encoder.cuda()
        from utils.prompt_conditioning import encode_prompt_blocks
        encoded = encode_prompt_blocks(self.pipe.text_encoder, [prompts], 1)
        self.pipe.text_encoder.cpu()
        self.pipe.generator.cuda()
        # Reuse the encoded contexts without keeping UMT5 resident during denoising.
        import pipeline.causal_diffusion_inference as module
        original = module.encode_prompt_blocks
        module.encode_prompt_blocks = lambda *_a, **_k: encoded
        try:
            noise = torch.randn(1, len(prompts) * self.chunk_size, 48, 44, 80,
                                device="cuda", dtype=torch.bfloat16)
            latent = self.pipe.inference(noise, [prompts], return_latents=True)
        finally:
            module.encode_prompt_blocks = original
        self.pipe.kv_cache_pos = self.pipe.kv_cache_neg = None
        self.pipe.crossattn_cache_pos = self.pipe.crossattn_cache_neg = None
        self.pipe.generator.cpu()
        torch.cuda.empty_cache()
        self.pipe.vae.cuda()
        video = self.pipe.vae.decode_to_pixel(latent, use_cache=False)
        # VAE wrapper returns [-1,1], unlike pipeline.inference's [0,1] output.
        frames = ((video[0].permute(0, 2, 3, 1).float().clamp(-1, 1) + 1) * 127.5).byte().cpu()
        self.pipe.vae.model.clear_cache()
        self.pipe.vae.cpu()
        self.pipe.generator.cuda()
        return frames
