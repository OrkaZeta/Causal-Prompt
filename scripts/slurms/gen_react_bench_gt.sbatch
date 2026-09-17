#!/usr/bin/env bash
# Generate ReactBench test videos with Wan2.2-Lightning 4-step T2V + TS-Attn.
# One job processes all 1,000 records sequentially on one GPU. H100 jobs are
# limited to 24 hours; resubmitting this script resumes by skipping existing IDs.
# Submit with: sbatch scripts/slurms/gen_react_bench_gt.sbatch
# Fixed generation settings: 832x480, 16 fps, 81 frames, seed 42, no CPU offload.
#SBATCH --partition=H100,RTX6000PRO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=react-gt
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh

INPUT="${INPUT:-data/react_bench_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/react_bench/test/gt_video}"
BASE_MODEL="${BASE_MODEL:-/projects/hi-paris/ZiyiData/Models/Wan2.2-T2V-A14B-Diffusers}"
LIGHTNING_LORA="${LIGHTNING_LORA:-/projects/hi-paris/ZiyiData/Models/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0}"
SEED="${SEED:-42}"
TS_ATTN_RATIO="${TS_ATTN_RATIO:-0.25}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

mkdir -p scripts/slurms/logs "${OUTPUT_DIR}"

python -m causal_prompt.react_bench_generate \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model "${BASE_MODEL}" \
  --lightning-lora "${LIGHTNING_LORA}" \
  --ts-attn-ratio "${TS_ATTN_RATIO}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
