> Information for reference only. No modification allowed.

## Third-party source policy

Everything under `third_party/` is reference-only. Do not modify, format, patch,
or generate files inside any `third_party/` repository. Implement project-specific
changes in `causal_prompt/` or another first-party module instead, keeping the
third-party submodule pointers unchanged unless an explicit dependency update is
requested.

## Local Development

* `$HOME="/home/zliu"`
* CUDA 13.1 / 12.9
    * Switching by `cuda-use 12.9` or `cuda-use 13.1`
* NVIDIA GeForce RTX 2060

## Remote Cluster

### Public files

HuggingFace downloaded model folders : `/projects/hi-paris/ZiyiData/Models`

Soft link to the working direction before running if needed.

### Cluster Connection

```text
ziyliu-24@gpu-gw.enst.fr
```

* `$HOME="/home/ids/ziyliu-24"`

---

## Slurm Partition Resource Table

| Partition  |  Max Runtime | Node Conf         | Nodes | GPUs per Node | Total GPUs |
|------------|-------------:|-------------------|------:|--------------:|-----------:|
| `A100`     | `1-00:00:00` | Standard A100     |    10 |             3 |         30 |
| `A100`     | `1-00:00:00` | Large A100        |     1 |             8 |          8 |
| `A30`      | `4-00:00:00` | Dual-GPU A30      |     1 |             2 |          2 |
| `A40`      | `4-00:00:00` | Dual-GPU A40      |    15 |             2 |         30 |
| `A40`      | `4-00:00:00` | Single-GPU A40    |     1 |             1 |          1 |
| `L40S`     | `1-00:00:00` | Quad-GPU L40S     |     5 |             4 |         20 |
| `L40S`     | `1-00:00:00` | Eight-GPU L40S    |     1 |             8 |          8 |
| `audible`  | `1-00:00:00` | Quad-GPU          |     1 |             4 |          4 |
| `H100`     | `1-00:00:00` | Dual-GPU H100     |     1 |             2 |          2 |
| `H100`     | `1-00:00:00` | Quad-GPU H100     |     1 |             4 |          4 |
| `CPU`      | `4-00:00:00` | CPU-only          |    10 |             0 |          0 |
| `cpu-high` | `5-00:00:00` | High-capacity CPU |     2 |             0 |          0 |

---

## GPU Slurm Job Template

```bash
#!/usr/bin/env bash
#SBATCH --partition=H100,audible,A100,L40S,A40,A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=JOB_NAME
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euxo pipefail

# ====== Setup ======
mkdir -p scripts/slurms/logs

# ====== Environment Info ======
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Allocated GPUs: ${SLURM_GPUS:-0}"

# ====== Module Loading ======
module purge
# remote cluster only supports cuda129 and installed flash-attn with torch-cu129
module load cuda/12.9
module load ffmpeg/8.1
module load uv/0.9.9

# ====== Verification ======
which ffmpeg
ffmpeg -version
which uv
uv --version
nvidia-smi || true

# ====== Run Workload ======
srun YOUR_COMMAND_HERE
```

Before submission:

```bash
mkdir -p scripts/slurms/logs
sbatch scripts/slurms/SCRIPT_NAME.slurm
```

- Register `causal_prompt` as a package in uv env for easier module import and use.
