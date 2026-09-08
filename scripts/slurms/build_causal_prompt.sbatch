#!/usr/bin/env bash
# Build the complete JSONL independently, then make training depend on this job:
# PREP_JOB=$(sbatch --parsable scripts/slurms/prepare_activitynet_videos.sbatch)
# BUILD_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" scripts/slurms/build_causal_prompt.sbatch)
# sbatch --dependency="afterok:${BUILD_JOB}" --export=ALL,EXP_ID=actnet-v1-seed42,SCHEDULE=causal scripts/slurms/exp1_cp_dmd_adapt.sbatch
#SBATCH --partition=H100,A100,audible,L40S,A40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=build-causal-prompt
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh

DATA_ROOT="${DATA_ROOT:-/projects/hi-paris/ZiyiData/Datasets/ActivityNet_Captions}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/Activity_Videos}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
QWEN_MODEL="${QWEN_MODEL:-/projects/hi-paris/ZiyiData/Models/Qwen3-VL-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p scripts/slurms/logs "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Starting causal-prompt dataset build"
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$(pwd)"
echo "video_root=${VIDEO_ROOT}"
echo "output_dir=${OUTPUT_DIR} files=activitynet_causal_5s_{train,test}.jsonl"
echo "model=${QWEN_MODEL} attention=auto batch_size=${BATCH_SIZE}"
echo "max_new_tokens=128 max_image_side=672 caption_retries=2 progress=every_100_records log_level=INFO"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

args=(
  --video-root "${VIDEO_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "${QWEN_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --device cuda:0
)

srun --unbuffered "${PYTHON}" -m causal_prompt.prompt.build_causal_prompt "${args[@]}"
for split in train test; do
  output="${OUTPUT_DIR}/activitynet_causal_5s_${split}.jsonl"
  test -s "${output}"
  echo "[$(date --iso-8601=seconds)] ${split}: $(wc -l < "${output}") records in ${output}"
done
