#!/usr/bin/env bash
# Generate videos from adapted causal-prompt checkpoints for any experiment.
# EXP_ID required : actnet-s3cf4-seed42, singleobj-s3cf4-seed42
# MODE required : [causal, current]
# EXP_NAME optional : experiment directory name, default exp1_cp_dmd
# DATA_ID optional : [single_obj, single_obj_v2, activitynet_causal], default activitynet_causal
# SPLIT optional : [train, test], default test
# SEEDS optional : non-negative int list separated by comma/colon/space, default 0
# STEPS optional : checkpoint-step int list separated by comma/colon/space, default latest
# MAX_SAMPLES optional : positive int, default all
# BATCH_SIZE optional : positive int, default 1
# CONFIG_PATH optional : override <run>/<mode>/resolved_config.yaml
# BASE_MODEL_PATH optional : override base_model_path in the resolved training config
# OUTPUT_ROOT optional : output root, default outputs
# OVERWRITE optional : [0, 1], default 0
# DATASET_PATH optional : custom JSONL path; required for a custom DATA_ID
# SAMPLE_ID_GLOBS optional : space-separated sample-ID globs
# SAMPLE_ID_REGEXES optional : space-separated full-match sample-ID regexes
# Examples:
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,MODE=causal,SEEDS=0,STEPS="1000 2000 3000",MAX_SAMPLES=10,DATA_ID=activitynet_causal scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,MODE=current,SEEDS=0,STEPS="1000 2000 3000",MAX_SAMPLES=10,DATA_ID=activitynet_causal scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,MODE=causal,SEEDS=0,STEPS="1000 2000 3000",MAX_SAMPLES=10,DATA_ID=single_obj scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=actnet-s3cf4-seed42,MODE=current,SEEDS=0,STEPS="1000 2000 3000",MAX_SAMPLES=10,DATA_ID=single_obj scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobjv2-s3cf4-seed42,MODE=causal,DATA_ID=single_obj_v2,DATASET_PATH=data/eval_single_obj_v2.jsonl,SPLIT=all,STEPS=2000,SAMPLE_ID_GLOBS="00*1_* 00*2_*" scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobjv2-s3cf4-seed42,MODE=current,DATA_ID=single_obj_v2,DATASET_PATH=data/eval_single_obj_v2.jsonl,SPLIT=all,STEPS=2000,SAMPLE_ID_GLOBS="00*1_* 00*2_*" scripts/slurms/gen_cp_adapt.sbatch
# sbatch --export=ALL,EXP_ID=singleobjv2-colour-s3cf4-seed42,MODE=current,DATA_ID=single_obj_v2_colour,DATASET_PATH=data/eval_single_obj_v2_colour.jsonl,SPLIT=all,STEPS="1000 2000" scripts/slurms/gen_cp_adapt.sbatch
#SBATCH --partition=H100,A100,audible,L40S,A40,A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gen-cp-adapt
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh

: "${EXP_ID:?Set EXP_ID to the adapted training run ID}"
: "${MODE:?Set MODE=causal or MODE=current}"
case "${MODE}" in
  causal|current) ;;
  *) echo "Unsupported MODE=${MODE}; expected causal or current." >&2; exit 2 ;;
esac

EXP_NAME="${EXP_NAME:-exp1_cp_dmd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DATA_ID="${DATA_ID:-activitynet_causal}"
DATASET_PATH="${DATASET_PATH:-}"
SPLIT="${SPLIT:-test}"
case "${SPLIT}" in
  train|test) ;;
  all)
    if [[ -z "${DATASET_PATH}" ]]; then
      echo "SPLIT=all requires an explicit DATASET_PATH." >&2
      exit 2
    fi
    ;;
  *) echo "Unsupported SPLIT=${SPLIT}." >&2; exit 2 ;;
esac
case "${DATA_ID}" in
  single_obj) DATASET_PATH="${DATASET_PATH:-data/eval_single_obj_${SPLIT}.jsonl}" ;;
  activitynet_causal) DATASET_PATH="${DATASET_PATH:-data/activitynet_causal_5s_${SPLIT}.jsonl}" ;;
  *)
    if [[ -z "${DATASET_PATH}" ]]; then
      echo "Unknown DATA_ID=${DATA_ID}; provide DATASET_PATH for custom data." >&2
      exit 2
    fi
    ;;
esac
for identifier in "${EXP_NAME}" "${EXP_ID}" "${DATA_ID}"; do
  if [[ ! "${identifier}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Experiment and data IDs must be filename-safe; received ${identifier}." >&2
    exit 2
  fi
done
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset does not exist: ${DATASET_PATH}" >&2
  exit 2
fi

SEEDS="${SEEDS:-0}"
seed_spec="${SEEDS//[,:]/ }"
read -r -a SEED_VALUES <<< "${seed_spec}"
if (( ${#SEED_VALUES[@]} == 0 )); then
  echo "SEEDS must contain at least one non-negative integer." >&2
  exit 2
fi
for seed in "${SEED_VALUES[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "SEEDS must be separated by comma, colon, or spaces; received ${SEEDS}." >&2
    exit 2
  fi
done

args=(
  --dataset "${DATASET_PATH}"
  --mode "${MODE}"
  --exp-name "${EXP_NAME}"
  --exp-id "${EXP_ID}"
  --output_root "${OUTPUT_ROOT}"
  --split "${SPLIT:-test}"
  --batch_size "${BATCH_SIZE:-1}"
  --seed "${SEED_VALUES[@]}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then args+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ -n "${SAMPLE_ID_GLOBS:-}" ]]; then
  read -r -a sample_id_globs <<< "${SAMPLE_ID_GLOBS}"
  for pattern in "${sample_id_globs[@]}"; do
    args+=(--sample-id-glob "${pattern}")
  done
fi
if [[ -n "${SAMPLE_ID_REGEXES:-}" ]]; then
  read -r -a sample_id_regexes <<< "${SAMPLE_ID_REGEXES}"
  for pattern in "${sample_id_regexes[@]}"; do
    args+=(--sample-id-regex "${pattern}")
  done
fi
if [[ -n "${STEPS:-}" ]]; then
  step_spec="${STEPS//[,:]/ }"
  read -r -a STEP_VALUES <<< "${step_spec}"
  for step in "${STEP_VALUES[@]}"; do
    if [[ ! "${step}" =~ ^[0-9]+$ ]]; then
      echo "STEPS must be separated by comma, colon, or spaces; received ${STEPS}." >&2
      exit 2
    fi
  done
  args+=(--steps "${STEP_VALUES[@]}")
fi
if [[ -n "${CONFIG_PATH:-}" ]]; then args+=(--config_path "${CONFIG_PATH}"); fi
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then args+=(--base_model_path "${BASE_MODEL_PATH}"); fi
if [[ "${OVERWRITE:-0}" == 1 ]]; then args+=(--overwrite); fi

run_dir="${OUTPUT_ROOT}/${EXP_NAME}/${EXP_ID}/${MODE}"
mkdir -p scripts/slurms/logs "${run_dir}"
export PYTHONUNBUFFERED=1
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "working_directory=$(pwd)"
  echo "exp_name=${EXP_NAME}"
  echo "exp_id=${EXP_ID}"
  echo "mode=${MODE}"
  echo "data_id=${DATA_ID}"
  echo "dataset_path=${DATASET_PATH}"
  echo "seeds=${SEEDS}"
  echo "steps=${STEPS:-latest}"
  echo
  echo "===== environment ====="
  env | sort
  echo
  echo "===== modules ====="
  module list 2>&1 || true
  echo
  echo "===== GPU ====="
  nvidia-smi || true
} > "${run_dir}/generation_env_${SLURM_JOB_ID}.txt"

srun --unbuffered "${PYTHON}" -m causal_prompt.cp_generate "${args[@]}"
