"""AlayaWorld v1.1 stage3: inference-only assembly of the upstream ViGeo rollout."""
from pathlib import Path
import sys
import torch


class AlayaWorldPipeline:
    fps = 24
    temporal_stride = 8
    chunk_size = 4
    height = 544
    width = 960
    input_mode = "i2v"

    def __init__(self, checkpoint: Path, models_root: Path):
        sys.path.insert(0, str(Path(__file__).with_name("vendor")))
        from alaya.config.loader import load_config
        from alaya.trainer.rollout_trainer import RolloutTrainer
        from alaya.model import loader
        from alaya.model.components import ModelComponents
        from alaya.memory.builder import build_history_encoder
        from alaya.model.lora import LoRAForwardManager
        from fastvideo.ltx2_streaming_vae import StreamingVAEEncoder
        import safetensors.torch as st
        cfg = load_config(Path(__file__).with_name("inference.yaml"))
        cfg.paths.base_transformer = str(models_root / "LTX-2.3/ltx-2.3-22b-dev.safetensors")
        cfg.paths.vae = cfg.paths.base_transformer
        cfg.paths.gemma = str(models_root / "gemma-3-12b-it-qat-q4_0-unquantized")
        cfg.paths.resume_checkpoint = str(models_root / "AlayaWorld-v1.1-stage2b")
        cfg.paths.dmd_resume = str(checkpoint)
        cfg.paths.history_encoder = str(checkpoint / "history_encoder.pt")
        cfg.spatial_memory.vigeo_checkpoint = str(models_root / "ViGeo1.1")
        cfg.spatial_memory.vigeo_repo_path = str(models_root / "ViGeo-code")
        cfg.runtime.gradient_checkpointing = False
        cfg.runtime.fsdp = False
        cfg.runtime.vae_decode_chunk_latents = 8
        cfg.runtime.precache_text_embeds = False
        cfg.runtime.vae_latent_cache_dir = None
        cfg.runtime.text_embed_cache_dir = None
        cfg.lora.train = False
        cfg.validation.cfg_scale = 1.0
        cfg.validation.stg_scale = 0.0
        # Do not construct the training optimizer, score model, critic, or data loader.
        self.runner = RolloutTrainer(cfg)
        self.runner._validation_cfg_scale = lambda: float(cfg.validation.cfg_scale)
        device, dtype = torch.device("cuda"), torch.bfloat16
        transformer = loader.load_transformer(cfg.paths.base_transformer, cfg, device=torch.device("cpu"), dtype=dtype)
        manager = LoRAForwardManager(trainable=False)
        manager.init_for_training(transformer, cfg.lora.targets, cfg.lora.rank, cfg.lora.alpha,
                                  dtype=dtype, device=torch.device("cpu"))
        lora_file = checkpoint / "lora.safetensors"
        state = st.load_file(str(lora_file), device="cpu")
        expected = manager.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"AlayaWorld LoRA keys differ: missing={set(expected)-set(state)}, unexpected={set(state)-set(expected)}")
        count = manager.load(str(lora_file))
        manager.register_hooks(transformer)
        manager.enable()
        print(f"[WM-AlayaWorld] LoRA strict loaded={count} missing=0 unexpected=0", flush=True)
        del state, expected
        raw_encoder, decoder = loader.load_vae(cfg.paths.vae, device=device, dtype=dtype)
        encoder = StreamingVAEEncoder(raw_encoder, device=device, dtype=dtype)
        text_encoder, encode_text = loader.load_text_encoder(cfg.paths.base_transformer, cfg.paths.gemma,
                                                            device=torch.device("cpu"), dtype=dtype)
        history = build_history_encoder(cfg.memory, in_channels=transformer.patchify_proj.in_features,
                                        out_channels=transformer.patchify_proj.out_features,
                                        device=device, dtype=dtype, checkpoint_path=cfg.paths.history_encoder)
        if cfg.memory.use_lr_branch:
            history.setup_lr_proj_from_patchify(transformer.patchify_proj)
        self.runner.history_encoder = history.eval()
        self.runner.components = ModelComponents(transformer.eval().requires_grad_(False), encoder,
                                                 decoder.eval(), text_encoder, encode_text, manager)
        self.cfg = cfg
        self.mode = cfg.validation.modes["custom_i2v"]

    @torch.inference_mode()
    def generate(self, prompts, seed, image=None):
        if image is None:
            raise ValueError("AlayaWorld requires an initial-state image")
        import numpy as np
        import random
        from PIL import Image, ImageOps
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        runner, cfg, components = self.runner, self.cfg, self.runner.components
        components.transformer.cpu()
        torch.cuda.empty_cache()
        components.text_encoder.cuda()
        contexts = {}
        for prompt in dict.fromkeys(prompts):
            encoded = components.encode_text(components.text_encoder, [prompt])
            context = encoded[0][0] if encoded[0].dim() == 3 else encoded[0]
            contexts[prompt] = context.to(device="cuda", dtype=torch.bfloat16)
        components.text_encoder.cpu()
        torch.cuda.empty_cache()
        components.transformer.cuda()
        pixels = torch.from_numpy(np.array(ImageOps.fit(Image.open(image).convert("RGB"), (self.width, self.height))))
        pixels = pixels.permute(2, 0, 1).float().div(127.5).sub(1)
        N = runner._validation_history_latents(self.mode)
        cond_end = runner._validation_cond_end(self.mode, self.chunk_size)
        prefix = runner._vigeo_target_prefix_pixel_frames(history_latent_frames=N)
        video = pixels.unsqueeze(0).repeat(prefix, 1, 1, 1)
        camera = torch.eye(4).repeat(prefix + len(prompts)*32 + 32, 1, 1)
        intrinsic = torch.tensor([[0.5,0,0.5],[0,0.5,0.5],[0,0,1.0]])
        metadata = {"video_id": Path(image).stem, "source": "custom_i2v", "has_camera": True,
                    "has_real_intrinsic": False, "cam_c2w": camera, "cam_c2w_raw": camera.clone(),
                    "intrinsic": intrinsic, "intrinsic_raw": intrinsic.clone(),
                    "pose_orig_w": float(self.width), "pose_orig_h": float(self.height), "frame_start": 0}
        target_start = cfg.layout.sink_latent_frames + N
        latent = runner._build_vigeo_validation_latent_full(video_pixels=video, metadata=metadata,
                    required_latents=target_start+len(prompts)*self.chunk_size, target_base_start=target_start,
                    history_latent_frames=N, allow_short=True, allow_empty_target=True)
        result = runner._validate_rollout_sample(video_pixels=video, latent_full=latent,
                    context=contexts[prompts[0]], scheduled_contexts=[contexts[p] for p in prompts],
                    scheduled_prompt_captions=prompts, negative_context=None, metadata=metadata,
                    mode_cfg=self.mode, K=self.chunk_size, rounds=len(prompts), N=N, gap_steps=0, cond_end=cond_end)
        frames = runner._decode_i2v_rollout_chunks_to_video_frames(nearby_latents=result[2], chunks=result[1])
        # ViGeo recurrent state must never leak between dataset samples.
        runner.vigeo_geometry = None
        torch.cuda.empty_cache()
        return frames
