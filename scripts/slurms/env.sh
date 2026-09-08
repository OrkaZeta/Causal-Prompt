#!/usr/bin/env bash

# Source this from SLURM jobs after `cd` to the repository root.
# Do not use `uv run` in SLURM jobs after installing compiled extensions such as
# flash-attn into .venv; uv may recreate the environment and remove them.

# ====== Module Loading ======
module purge
# The remote cluster only supports CUDA 12.9; FlashAttention is installed
# against the corresponding CUDA-enabled PyTorch build.
module load cuda/12.9 || module load cuda || echo "CUDA load failed"
nvcc --version || echo "nvcc not found"

module load ffmpeg/8.1 || echo "ffmpeg load failed"
which ffmpeg
ffmpeg -version

module load uv/0.9.9
which uv
uv --version

# ====== Verification ======
nvidia-smi || true

# ====== Virtual Environment ======
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python at ${PYTHON}." >&2
  echo "Create/sync the environment before submitting jobs, then install flash-attn if needed:" >&2
  echo "  uv sync" >&2
  echo "  . ${VENV_DIR}/bin/activate" >&2
  echo "  python -m pip install flash-attn --no-build-isolation" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

echo "venv=${VIRTUAL_ENV}"
python - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version.split()[0])
PY
