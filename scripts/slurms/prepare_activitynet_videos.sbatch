#!/usr/bin/env bash
# Extract the multipart ActivityNet archive once, before VLM captioning.
#SBATCH --partition=CPU,cpu-high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=prepare-activitynet
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:-/projects/hi-paris/ZiyiData/Datasets/ActivityNet_Captions}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/Activity_Videos}"

mkdir -p scripts/slurms/logs
if find "${VIDEO_ROOT}" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mkv' \) -print -quit 2>/dev/null | grep -q .; then
  echo "ActivityNet videos are already extracted under ${VIDEO_ROOT}."
  exit 0
fi

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"
parts=(ActivityNet_Videos.tar.part-*)
if [[ ! -f "${parts[0]}" ]]; then
  echo "Multipart ActivityNet archive not found under ${DATA_ROOT}." >&2
  exit 2
fi
cat "${parts[@]}" | tar -xf -

if ! find "${VIDEO_ROOT}" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mkv' \) -print -quit | grep -q .; then
  echo "Archive extraction produced no videos under ${VIDEO_ROOT}." >&2
  exit 2
fi
echo "ActivityNet videos prepared under ${VIDEO_ROOT}."
