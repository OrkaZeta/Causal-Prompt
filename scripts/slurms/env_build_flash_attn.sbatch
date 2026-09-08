#!/usr/bin/env bash
#SBATCH --partition=H100,audible,A100,L40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=build-flash-attn
#SBATCH --output=slurms/logs/%x-%j.out
#SBATCH --error=slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euxo pipefail

# ====== Setup ======
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurms/logs

# ====== Environment Info ======
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Allocated GPUs: ${SLURM_GPUS:-0}"

source slurms/env.sh

# ====== Run Workload ======
export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS="$((SLURM_CPUS_PER_TASK / 2))"
export NVCC_THREADS=1

uv sync --frozen --python 3.10

uv pip uninstall --python "${PYTHON}" flash-attn || true
uv pip install --python "${PYTHON}" setuptools wheel
uv pip install \
  --python "${PYTHON}" \
  --no-build-isolation \
  --no-cache-dir \
  "flash_attn"

python - <<'PY'
import flash_attn
import torch

print("flash-attn:", flash_attn.__version__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
