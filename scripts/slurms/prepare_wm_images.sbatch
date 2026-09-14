#!/usr/bin/env bash
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=wm-initial-images
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh
srun --unbuffered "${PYTHON}" -m causal_prompt.prepare_wm_images \
  --dataset "${DATASET_PATH:-data/eval_single_obj_v2_colour.jsonl}" \
  --save-dir "${WM_IMAGE_DIR:-outputs/zeroshot_inputs/single_obj_v2_colour/initial_images}" \
  --split train --seed 0
