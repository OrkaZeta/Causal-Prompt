#!/usr/bin/env bash
# Generate ReactBench test reference videos from each row's "Full Prompt".
# One job processes all 1,000 records sequentially on one GPU. H100 jobs are
# limited to 24 hours; resubmitting this script resumes by skipping existing IDs.
# Submit with: sbatch scripts/slurms/gen_react_bench_gt.sbatch
# Fixed generation settings: 1280x720, 16 fps, 81 frames, seed 42, no CPU offload.
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
TASK="${TASK:-t2v-14B}"
CHECKPOINT="${CHECKPOINT:-/projects/hi-paris/ZiyiData/Models/Wan2.1-T2V-14B}"
SEED="${SEED:-42}"
COMPILE_MODE="${COMPILE_MODE:-default}"

mkdir -p scripts/slurms/logs "${OUTPUT_DIR}"

python -m causal_prompt.react_bench_generate \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --num-shards 1 \
  --shard-index 0 \
  --torch-compile \
  --torch-compile-mode "${COMPILE_MODE}"
