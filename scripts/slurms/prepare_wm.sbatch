#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=prepare-wm
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${MODEL:?Set WM-LongLive or WM-AlayaWorld}"
module load uv/0.9.9
# A private overlay keeps CF's existing torch/flash-attn and transformers untouched.
base_python="${PWD}/.venv/bin/python"
wm_venv="${PWD}/.venv-wm"
exec 9>"${PWD}/.wm-environment.lock"
flock 9
if [[ ! -f "${wm_venv}/.ready" ]]; then
  uv venv --python "${base_python}" "${wm_venv}" --allow-existing
  "${base_python}" - "${wm_venv}" <<'PY'
import site,sys
from pathlib import Path
v=Path(sys.argv[1]); dst=next(v.glob('lib/python*/site-packages'))
(dst/'base_environment.pth').write_text('\n'.join(site.getsitepackages())+'\n')
PY
  # Pin only the incompatible API layer; use the existing CUDA torch build.
  uv pip install --python "${wm_venv}/bin/python" --no-deps \
    'transformers==4.57.6' 'huggingface-hub==0.36.2' 'decord==0.6.0' 'timm==1.0.22'
  touch "${wm_venv}/.ready"
fi
flock -u 9
"${wm_venv}/bin/python" scripts/download_wm_models.py --model "${MODEL}"
if [[ "${MODEL}" == WM-AlayaWorld ]]; then
  vigeo="/projects/hi-paris/ZiyiData/Models/ViGeo-code"
  if [[ ! -d "${vigeo}/.git" ]]; then
    git clone --depth 1 https://github.com/aigc3d/ViGeo.git "${vigeo}"
  fi
  git -C "${vigeo}" rev-parse HEAD > "${vigeo}/inference_revision.txt"
fi
