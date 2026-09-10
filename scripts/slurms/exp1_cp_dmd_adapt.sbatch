#!/usr/bin/env bash
# Submit both matched arms separately:
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,SCHEDULE=causal scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,SCHEDULE=current scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobj-s3cf4-seed42,SCHEDULE=causal,DATASET=data/eval_single_obj_train.jsonl,MAX_STEPS=2000 scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobj-s3cf4-seed42,SCHEDULE=current,DATASET=data/eval_single_obj_train.jsonl,MAX_STEPS=2000 scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobjv2-s3cf4-seed42,SCHEDULE=causal,DATASET=data/eval_single_obj_v2.jsonl,MAX_STEPS=2000 scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobjv2-s3cf4-seed42,SCHEDULE=current,DATASET=data/eval_single_obj_v2.jsonl,MAX_STEPS=2000 scripts/slurms/exp1_cp_dmd_adapt.sbatch
# --nodelist=nodeaudible02,node53,nodesumo01
# sbatch --partition=H100 --gpus=1 --export=ALL,EXP_ID=actnet-s3cf4-seed42,SCHEDULE=causal,NPROC_PER_NODE=1 scripts/slurms/exp1_cp_dmd_adapt.sbatch
# sbatch --partition=L40S,A100 --gpus=3 --export=ALL,EXP_ID=actnet-s3cf4-seed42,SCHEDULE=current,DATASET=data/eval_single_obj_train.jsonl,MAX_STEPS=2000,NPROC_PER_NODE=3  scripts/slurms/exp1_cp_dmd_adapt.sbatch
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=140G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=exp1-cp-dmd
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh

: "${SCHEDULE:?Set SCHEDULE=current or SCHEDULE=causal}"
: "${EXP_ID:?Set a human-readable EXP_ID, for example actnet-s3cf4-seed42}"
case "${SCHEDULE}" in
  current|causal) ;;
  *) echo "SCHEDULE must be current or causal; received ${SCHEDULE}." >&2; exit 2 ;;
esac

DATASET="${DATASET:-data/activitynet_causal_5s_train.jsonl}"
CHECKPOINT="${CHECKPOINT:-/projects/hi-paris/ZiyiData/Models/Causal-Forcing/framewise/causal_forcing.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MAX_STEPS="${MAX_STEPS:-5000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${DATASET}" in *.jsonl) ;; *) echo "DATASET must be JSONL." >&2; exit 2 ;; esac
if [[ ! -s "${DATASET}" ]]; then
  echo "Missing causal-prompt JSONL: ${DATASET}. Run build_causal_prompt.sbatch first." >&2
  exit 2
fi

mkdir -p scripts/slurms/logs
srun "${PYTHON}" -m causal_prompt.run_cp_dmd_adapt \
  --schedule "${SCHEDULE}" \
  --exp-id "${EXP_ID}" \
  --dataset "${DATASET}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-steps "${MAX_STEPS}" \
  --nproc-per-node "${NPROC_PER_NODE}"
