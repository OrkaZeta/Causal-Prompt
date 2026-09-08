## Causal Forcing series baseline

- Hardware : NVIDIA H100 * 1

### Datasets

- **VidProM** : prompts
- **OpenVid** : paired Text caption - Video Clip
- Auto regressive Diffusion Training : Causal Forcing 6K toy dataset (in LMDB shard)

    `hf download zhuhz22/Causal-Forcing-data --local-dir dataset`

Files save to `/projects/hi-paris/ZiyiData/Datasets/ActivityNet_Captions`

### Train

```yaml
lr: 2e-6
opt: Adam
- beta1:0.0
- beta2:0.999
batch: 1
```

| Layout          | causal units | latent frames / block | Total latent frames | Total RGB frames |
|-----------------|--------------|-----------------------|---------------------|------------------|
| Chunk-wise / CW | 7            | 3                     | 21                  | 81 (16 FPS, 5s)  |
| Frame-wise / FW | 21           | 1                     | 21                  | 81 (16 FPS, 5s)  |

### Stage 1：AR Diffusion Training

$\left.\begin{aligned}
\text{CF : }\textbf{VidProM} + \text{Wan2.1 1.3B}\\
\text{CF++ : }\textbf{OpenVid}\left\{\begin{aligned}\text{raw video}\\\text{caption}\end{aligned}\right.
\end{aligned}\right\}$$\rightarrow$ video-caption pairs $\rightarrow$ Wan VAE encode $\rightarrow$ AR Diffusion (init from Wan2.1-T2V-1.3B)

Teacher Forcing : $p(x_0^i\mid x_0^{\lt i}, c)$, i.e. $\scriptstyle{\left.\begin{aligned}
\text{GT history}\\
\text{Current noisy latent}\\
\text{Current timestamp}\\
\text{Condition}
\end{aligned}\right\}\rightarrow\text{Current flow velocity}}$

> **Wan VAE** temporal relations : $T_\text{latent} = \frac{T_\text{video} - 1}{4} + 1$, with 16 FPS.
>
> | **16 FPS Dur** | 0 s | 1 s | 2 s | 3 s | 4 s | 5 s |
> |----------------|-----|-----|-----|-----|-----|-----|
> | **RGB**        | 1   | 17  | 33  | 49  | 65  | 81  |
> | **Latent**     | 1   | 5   | 9   | 13  | 17  | 21  |

### Stage 2：Causal ODE Initialisation (Few-step accelerate)

S1 AR diffusion teacher $\xrightarrow{PF-ODE}$ ODE trajectory / flow-map pairs $(x_{t_a},t_a)\rightarrow x_{t_b}\rightarrow$ Few-step 1.3B Student

### Stage 3：Asymmetric DMD (Fix path)

$\text{S2 Causal Student}\xrightarrow{\text{self-rollout}} x_{\text{fake}}
\left\{\begin{aligned}
\xrightarrow{\text{Bid Teacher (Wan21 14B)}} s_\text{real}(x_t,t)\\
\xrightarrow{\text{Causal Score Model (from S2)}} s_\text{fake}(x_t,t)
\end{aligned}\right\}
\xrightarrow{\Delta_{\text{DMD}}\propto s_{\text{real}}-s_{\text{fake}}}$ update causal 1.3B student & causal 1.3B score model

---

## Exp 0 : Zero-shot Baseline with Causal Prompt design

Causal-Forcing weight for inference and eval.

Prompt Schedule modes:

| Mode           | ID               | Visibility                                      | $\mathrm{Lat}(F_0)$                   | $\mathrm{Lat}(\{F_{e_1}\})$         | $\mathrm{Lat}(\{F_{e_k}\})$         | $\mathrm{Lat}(\{F_{e_{-1}}\})$      |
|----------------|------------------|-------------------------------------------------|---------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|
| Full (default) | `full`           | $t_0 \rightarrow t_T$                           | $G+E_{1:N}$                           | $G+E_{1:N}$                         | $G+E_{1:N}$                         | $G+E_{1:N}$                         |
| Full-Tagged    | `full_tagged`    | [past / current / future] $t_0 \rightarrow t_T$ | $G+[\mathrm{tag}^{(0)}_{1:N}]E_{1:N}$ | $[\mathrm{tag}^{(1)}_{1:N}]E_{1:N}$ | $[\mathrm{tag}^{(k)}_{1:N}]E_{1:N}$ | $[\mathrm{tag}^{(N)}_{1:N}]E_{1:N}$ |
| Causal Next    | `causal_next`    | $t_0 \rightarrow t_{x+1}$                       | $G+E_1$                               | $E_{1:2}$                           | $E_{1:k+1}$                         | $E_{1:N}$                           |
| Causal         | `causal`         | $t_0 \rightarrow t_x$                           | $G$                                   | $E_1$                               | $E_{1:k}$                           | $E_{1:N}$                           |
| Current Future | `current_future` | $t_x \rightarrow t_T$                           | $G+E_{1:N}$                           | $E_{1:N}$                           | $E_{k:N}$                           | $E_N$                               |
| Current Next   | `current_next`   | $t_x \rightarrow t_{x+1}$                       | $G+E_1$                               | $E_{1:2}$                           | $E_{k:k+1}$                         | $E_N$                               |
| Current        | `current`        | $t_x$                                           | $G$                                   | $E_1$                               | $E_k$                               | $E_N$                               |

Models Groups :

| Group | desc                     | Models                         | ID Avail                               | Prompt         |
|-------|--------------------------|--------------------------------|----------------------------------------|----------------|
| B0    | Bidirectional Teachers   | Wan 2.1, Wan 2.2               | B0-Wan21, B0-Wan22                     | `full` only    |
| T1    | Diffusion AR Teachers    | AR Diffusion                   | T1-FW, T1-CW                           | All mode avail |
| S2    | Causal ODE/CD Students   | Causal ODE, Causal CD          | S2-FW/CW-ODE, S2-FW/CW-CD              | All mode avail |
| S3    | Causal Few-Step Students | CF (4-steps), CFPP (2/1-steps) | S3-FW/CW-CF4, S3-FW-CFPP2, S3-FW-CFPP1 | All mode avail |
- Currently only implement and use B0-Wan21, B0-Wan22, T1-FW, S3-FW-CF4, S3-FW-CFPP2

All released zero-shot families use one SLURM entrypoint. Each job selects one
model and schedule, then generates every selected JSONL data ID for every seed.
B0 Wan models accept only `full`.

```bash
sbatch --export=ALL,MODEL=B0-wan21,SCHEDULE=full,DATA_ID=single_obj,SEEDS=0:1 \
  scripts/slurms/gen_zeroshot.sbatch
sbatch --export=ALL,MODEL=S3-FW-CF4STEP,SCHEDULE=causal,\
DATA_ID=activitynet_causal,SEEDS=0:1 \
  scripts/slurms/gen_zeroshot.sbatch
```

---

## Exp 1 : DMD-Adapted Causal Forcing with Causal Prompt

Causal Prompt schedules `current` and `causal`, initialized from the released
4-step frame-wise Causal Forcing weight.

### Hypothesis and controlled arms

The causal schedule should improve event ordering and reduce future-event leakage
relative to current-only conditioning, while current-only conditioning may retain
better local event fidelity. Both arms use identical data order, seed, optimizer,
DMD update ratio, number of steps, and initialization.

| Arm | Generator conditioning at latent block k | DMD real/fake score conditioning |
|---|---|---|
| `current` | initial view at block 0; active event(s) afterward | full initial view + all events |
| `causal` | initial view at block 0; all started event(s) afterward | full initial view + all events |

The score-side policy is deliberately asymmetric. The causal generator is called
one latent block at a time and can refresh cross-attention when its prompt changes.
The bidirectional Wan score models process the complete noisy clip in one call and
therefore receive one full-video prompt. This keeps the DMD target architecture
unchanged and isolates adaptation of the causal student.

### Fixed training contract

- Student initialization: `S3-FW-CF4STEP` EMA checkpoint.
- Video latent layout: 21 frame-wise blocks, 16 channels, 60 × 104 spatial latent.
- Denoising steps: `[1000, 750, 500, 250]` with warped Wan timesteps.
- DMD: Wan2.1-T2V-14B real score, Wan2.1-T2V-1.3B fake score, CFG 3.0.
- Optimizers: generator LR `2e-6`, fake-score LR `4e-7`, AdamW betas `(0, 0.999)`.
- Update ratio: one generator update per five fake-score updates.
- Batch size 1, BF16, EMA 0.99 after step 200, 5,000 adaptation steps, seed 42.
- Hardware target: one 80 GB H100. Each arm writes to an independent output folder.

### Implementation and execution

- `causal_prompt/prompt/build_causal_prompt.py` deterministically selects the longest
  valid event sequence (earliest on ties), decodes the selected crop's initial
  frame, captions it with Qwen3-VL-8B-Instruct, and writes each complete record
  directly to the single final causal-prompt JSONL. There is no separate
  initial-view manifest. Batched decoder-only generation uses left padding; an
  empty batch response is retried per image before the build fails. The resumable
  builder shows annotation/caption progress bars and emits timestamped progress
  every 100 newly written records by default.
- `causal_prompt/prompt/training.py` validates the 81-RGB/21-latent contract,
  preserves sample-major schedules during collation, and deduplicates text encoding.
- The self-forcing training rollout switches conditioning at block boundaries and
  invalidates only cross-attention caches; temporal self-attention history remains.
- `causal_prompt/models/causal_forcing/configs/exp1_cp_dmd_framewise.yaml`
  contains shared hyperparameters.
- `causal_prompt/run_cp_dmd_adapt.py` preflights the entire JSONL before launching.
- Every run stores metrics at
  `outputs/exp1_cp_dmd/<exp_id>/<schedule>/tensorboard/`, checkpoints under
  `.../checkpoints/step_<step>/`, plus `resolved_config.yaml` and `run.json`.
  TensorBoard tags include generator/critic loss and gradient norm, DMD gradient
  norm, sampled score/critic timestep, both learning rates, and iteration time.
  A non-empty run directory is rejected because checkpoint resume is not yet
  implemented; choose a new explicit `exp_id` instead. No path hash is generated.

```bash
# Extract the multipart archive, then build the complete training JSONL
sbatch scripts/slurms/prepare_activitynet_videos.sbatch
causal-prompt-build-dataset \
  --annotations-dir /projects/hi-paris/ZiyiData/Datasets/ActivityNet_Captions \
  --video-root /projects/hi-paris/ZiyiData/Datasets/ActivityNet_Captions/Activity_Videos \
  --model-name /projects/hi-paris/ZiyiData/Models/Qwen3-VL-8B-Instruct \
  --output-dir data

# CPU-only manifest/schedule check
python -m causal_prompt.run_cp_dmd_adapt \
  --exp-id actnet-v1-seed42 --schedule causal \
  --dataset data/activitynet_causal_5s_train.jsonl --dry-run

# Cluster runs
PREP_JOB=$(sbatch --parsable scripts/slurms/prepare_activitynet_videos.sbatch)
BUILD_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  scripts/slurms/build_causal_prompt.sbatch)
sbatch --dependency="afterok:${BUILD_JOB}" \
  --export=ALL,EXP_ID=actnet-v1-seed42,SCHEDULE=current \
  scripts/slurms/exp1_cp_dmd_adapt.sbatch
sbatch --dependency="afterok:${BUILD_JOB}" \
  --export=ALL,EXP_ID=actnet-v1-seed42,SCHEDULE=causal \
  scripts/slurms/exp1_cp_dmd_adapt.sbatch
```

Generate adapted checkpoints through the experiment-independent entrypoint.
`EXP_NAME` selects `outputs/<exp-name>/`; future Exp 2/3/4 runs do not require
another generation sbatch.

```bash
sbatch --export=ALL,EXP_NAME=exp1_cp_dmd,EXP_ID=actnet-v1-seed42,\
MODE=causal,SEEDS=0:1,STEPS=3000 \
  scripts/slurms/gen_cp_adapt.sbatch
```

Monitor a running build with `squeue -j "${BUILD_JOB}"` and
`tail -f scripts/slurms/logs/build-causal-prompt-${BUILD_JOB}.{out,err}`. The
batch script accepts `BATCH_SIZE` through `sbatch --export`. Caption generation
uses 128 output tokens, a 672-pixel maximum image side, two empty-caption retries,
  automatic attention selection, INFO logging, and a progress log every 100 records.
Re-submission automatically validates complete rows and appends only pending samples.
One run reads ActivityNet `train` and `val1`, then writes them into the same JSONL
as `train` and `test`. `val2` is unused. Training loads only the `train` rows.
### Evaluation plan

Evaluate the unadapted 4-step checkpoint plus both adapted arms on the same held-out
ActivityNet subset and the same seeds. Report event-caption alignment per annotated
interval, event-order accuracy, future-object/action leakage before each event start,
temporal consistency, and standard video quality. Use paired per-sample differences
with bootstrap 95% confidence intervals; do not select the final checkpoint on the
test subset. The implementation in this change covers data construction, scheduling,
training, and launch; metric-specific evaluators remain a separate experiment stage.

### ActivityNet Captions https://huggingface.co/datasets/friedrichor/ActivityNet_Captions

| split | videos | 2 Events | 3 Events | 4 Events | 5+ Events |
|-------|--------|----------|----------|----------|-----------|
| train | 10009  | 1936     | 4125     | 1891     | 2057      |
| val1  | 4917   | 1018     | 2171     | 890      | 838       |

| split  | train | val1 |
|--------|-------|------|
| videos | 10009 | 4917 |

```json
{"video_id": "v_*","video": "v_*.mp4","caption": "E1. E2","duration": 0.0,"timestamps": [[0.0, 0.0],[0.0, 0.0]],"sentences": ["E1", "E2"]}
```

#### Causal Prompt Dataset Build

1. **Filtering : 2+ sequence video, and** $T_\text{span}=t_\text{end}(E_j)-t_\text{start}(E_i)\le5s$

    Keep every maximal valid sequence. Overlapping sequences such as E1–E2 and
    E2–E3 are both retained when neither can be extended, while a shorter sequence
    is removed when it is contained by a longer valid sequence such as E1–E3.

        e.g. if both $E_1,E_2$ and $E_2,E_3$ satisfy the 5s constraint and both contain two events, select $(E_1,E_2)$

2. **5s Crop**

    Prefer to align the end of the last selected event to the end of the clip:

    $\mathrm{crop_{start}}=\max(0,t_\text{end}(E_j)-5)$
    $\mathrm{crop_{end}}=\mathrm{crop_{start}}+5$

    Therefore, whenever $t_\text{end}(E_j)\ge5s$, the final selected event satisfies $t_\text{end}^\text{local}(E_j)=5s$. If $t_\text{end}(E_j)<5s$, the clip starts from source-video time 0 and the remaining duration appears after the selected sequence. All timestamps are converted to the local timeline using $t_\text{local}=t_\text{original}-\mathrm{crop_{start}}$.

3. Video samples should represented as exactly **81 RGB frames over 5s @ 16 FPS**.

    Wan VAE converts : 81 RGB frames → 21 latent frames with temporal relation $T_\text{latent}=(T_\text{video}-1)/4+1$

4. Initial-view Condition : $\text{Filtered video}\xrightarrow{\mathrm{crop}}F_\text{init}\xrightarrow{\text{Qwen3-VL-8B-Instruct}}I$

    Video fetch from `ActivityNet_Videos.tar.part-{000,…,007}`; → *.mp4

    > Describe only what is visibly present in the image. Do not infer future actions, intentions, hidden objects, or events. Return a concise description containing: scene/environment, visible subjects, visible objects, camera viewpoint, and visual style/lighting.

    $I$ must not introduce future actions, future objects, intentions, event outcomes, or hidden states.

5. **Final Dataset:** Save train/test separately as `activitynet_causal_5s_train.jsonl` and `activitynet_causal_5s_test.jsonl`.

    ```json
    {"sample_id": "v_xxxxx_E2_E3","split": "train","video_id": "v_xxxxx","video": "v_xxxxx.mp4","crop_start": 37.25,"crop_end": 42.25,"fps": 16,"num_rgb_frames": 81,"num_latent_frames": 21,"initial_view": "A man is visible in an indoor room, viewed from a medium-wide camera with natural lighting.","num_events": 3,"events": [{"event_id": 2,"start": 0.75,"end": 2.10,"caption": "A man walks toward a table."},{"event_id": 3,"start": 1.95,"end": 3.40,"caption": "The man picks up an object."},{"event_id": 4,"start": 3.60,"end": 5.00,"caption": "The man walks away from the table."}]}
    ```


---

## Exp 2 : ODE + DMD with Causal Prompt design
