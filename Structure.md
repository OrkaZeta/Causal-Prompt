# Structure of code

## Scripts at the CausalPrompt

`causal_prompt` is an installable uv-managed package. Run `uv sync --frozen`
after cloning; modules and the `causal-prompt-dmd` and
`causal-prompt-build-dataset` commands are then available from the environment.

```text
./causal_prompt
├── models
│   ├── causal_forcing  # Move from CF/CF++ repo 
│   ├── cmd  # Move from Nvidia CMD repo
│   ├── wan21  # Move from Wan repo
│   └── wan22  # Move from Wan repo
├── cf_generate.py
├── wan_generate.py
├── cp_generate.py
└── run_cp_dmd_adapt.py

```


## Data related functions and utils

### Raw Data

`data/` contains artifacts only—JSONL, CSV, Markdown reports, videos, and model
inputs/outputs. Python builders and analysis code live in `causal_prompt/prompt/`.

```text
./data
├── eval_single_obj_train.jsonl
└── eval_single_obj_test.jsonl
└── 
```

### Prompt schedule

Offering the Datasets inherit class for model generating video / distillation input.

```text
./causal_prompt/prompt
├── analyze_activitynet.py
├── build_causal_prompt.py
├── schedule.py
└── training.py
```

```python
# For each video generation offering
@dataclass
class ScheduledPrompt:
    prompt_id: str
    prompt: str
    source: dict[str, Any]
    mode: str
    fps: float
    duration_sec: float
    pixel_frame_count: int
    latent_frame_count: int
    chunk_size: int
    events: tuple[TimedEvent, ...]
    pixel_prompts: list[str]
    frame_prompts: list[str]
    chunk_prompts: list[str]
```

### Outputs formats

1. For Zero-shot test, we will generate video using Wan2.1, Wan2.2
   outputs/zeroshot_baseline/single_obj/B0-wan22/full/vid/0001_colour_traffic_light_seed0.mp4


```text
.
├── causal_prompt
│   ├── Experiment.md
│   ├── Structure.md
│   ├── config
│   ├── eval
│   ├── models
│   │   ├── causal_forcing
│   │   ├── cmd
│   │   ├── wan21
│   │   └── wan22
│   ├── prompt
│   │   ├── analyze_activitynet.py
│   │   ├── build_causal_prompt.py
│   │   ├── schedule.py
│   │   └── training.py
│   ├── cp_generate.py
│   ├── run_cp_dmd_adapt.py
│   ├── cf_generate.py
│   └── wan_generate.py
├── data
│   ├── activitynet
│   ├── cf_baseline.txt
│   ├── eval_single_obj_train.jsonl
└── eval_single_obj_test.jsonl
│   └── sample.jsonl
├── outputs
│   └── zeroshot_baseline
│       └── single_obj
│           └── B0-wan{21,22}
│               └── full
│                   ├── env_961467.txt
│                   └── vid
│                       └── *.mp4
├── papers
├── scripts
│   └── slurms
│       ├── env.sh
│       ├── prepare_activitynet_videos.sbatch
│       ├── build_causal_prompt.sbatch
│       ├── exp1_cp_dmd_adapt.sbatch
│       ├── env_build_flash_attn.sbatch
│       ├── gen_zeroshot.sbatch
│       ├── gen_cp_adapt.sbatch
│       └── logs
│           ├── *.err
│           └── *.out
├── DevNote.md
├── Experiment.md
├── Structure.md
├── pyproject.toml
└── uv.lock
```
