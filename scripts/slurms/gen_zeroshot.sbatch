#!/usr/bin/env bash
# One zero-shot model/schedule/data job; every selected data ID and seed is generated.
# MODEL : [B0-wan21, B0-wan22, T1-FW, S3-FW-CF4STEP, S3-FW-CFPP2STEP]
# SCHEDULE : [full, full_tagged, causal_next, causal, current_future, current_next, current]
#            B0-wan21 and B0-wan22 support full only.
# DATA_ID : [single_obj, activitynet_causal]
# SPLIT optional : [train, test], default test
# SEEDS optional : non-negative int list separated by comma/colon/space, default 0
# MAX_SAMPLES optional : positive int, default all
# BATCH_SIZE optional : positive int, default 1
# OVERWRITE optional : [0, 1], default 0
# DATASET_PATH optional : custom JSONL path; required for a custom DATA_ID
# Examples:
# sbatch --export=ALL,MODEL=B0-wan21,SCHEDULE=full,DATA_ID=single_obj,SEEDS=0:1 scripts/slurms/gen_zeroshot.sbatch
# sbatch --export=ALL,MODEL=S3-FW-CF4STEP,SCHEDULE=causal,DATA_ID=activitynet_causal,SEEDS=0 scripts/slurms/gen_zeroshot.sbatch
#SBATCH --partition=H100,A100,audible,L40S,A40,A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gen-zeroshot
#SBATCH --output=scripts/slurms/logs/%x-%j.out
#SBATCH --error=scripts/slurms/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurms/env.sh

: "${MODEL:?Set MODEL to B0-wan21, B0-wan22, T1-FW, S3-FW-CF4STEP, or S3-FW-CFPP2STEP}"
: "${SCHEDULE:?Set SCHEDULE to a supported prompt schedule}"

case "${SCHEDULE}" in
  full|full_tagged|causal_next|causal|current_future|current_next|current) ;;
  *) echo "Unsupported SCHEDULE=${SCHEDULE}." >&2; exit 2 ;;
esac

DATA_ID="${DATA_ID:-single_obj}"
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
if [[ ! "${DATA_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "DATA_ID must be filename-safe; received ${DATA_ID}." >&2
  exit 2
fi
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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zeroshot_baseline}"
VIDEO_DIR="${OUTPUT_ROOT}/${DATA_ID}/${MODEL}/${SCHEDULE}/vid"
BATCH_SIZE="${BATCH_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p scripts/slurms/logs "${VIDEO_DIR}"
export PYTHONUNBUFFERED=1

run_dir="${OUTPUT_ROOT}/${DATA_ID}/${MODEL}/${SCHEDULE}"
env_file="${run_dir}/env_${SLURM_JOB_ID}.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "working_directory=$(pwd)"
  echo "model=${MODEL}"
  echo "schedule=${SCHEDULE}"
  echo "data_id=${DATA_ID}"
  echo "dataset_path=${DATASET_PATH}"
  echo "split=${SPLIT}"
  echo "seeds=${SEEDS}"
  echo "video_dir=${VIDEO_DIR}"
  echo
  echo "===== environment ====="
  env | sort
  echo
  echo "===== modules ====="
  module list 2>&1 || true
  echo
  echo "===== GPU ====="
  nvidia-smi || true
} > "${env_file}"

common_limit=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  common_limit=(--max_samples "${MAX_SAMPLES}")
fi

case "${MODEL}" in
  B0-wan21|B0-wan22)
    if [[ "${SCHEDULE}" != full ]]; then
      echo "${MODEL} only supports SCHEDULE=full." >&2
      exit 2
    fi
    if [[ "${MODEL}" == B0-wan21 ]]; then
      task="t2v-1.3B"
      checkpoint="${WAN21_CKPT_DIR:-/projects/hi-paris/ZiyiData/Models/Wan2.1-T2V-1.3B}"
      wan_extra=()
    else
      task="t2v-5B"
      checkpoint="${WAN22_CKPT_DIR:-/projects/hi-paris/ZiyiData/Models/Wan2.2-TI2V-5B}"
      wan_extra=(--convert_model_dtype)
    fi
    args=(
      --prompt "${DATASET_PATH}"
      --prompt_schedule full
      --task "${task}"
      --ckpt_dir "${checkpoint}"
      --save_dir "${VIDEO_DIR}"
      --frame_num 81
      --fps 16
      --split "${SPLIT}"
      --seed "${SEED_VALUES[@]}"
      --offload_model "${OFFLOAD_MODEL:-true}"
      --skip_existing
      "${common_limit[@]}"
      "${wan_extra[@]}"
    )
    if [[ "${T5_CPU:-1}" == 1 ]]; then args+=(--t5_cpu); fi
    if [[ "${OVERWRITE}" == 1 ]]; then
      filtered=()
      for value in "${args[@]}"; do
        [[ "${value}" == --skip_existing ]] || filtered+=("${value}")
      done
      args=("${filtered[@]}")
    fi
    srun --unbuffered "${PYTHON}" -m causal_prompt.wan_generate "${args[@]}"
    ;;
  T1-FW|S3-FW-CF4STEP|S3-FW-CFPP2STEP)
    args=(
      --model "${MODEL}"
      --dataset "${DATASET_PATH}"
      --prompt_schedule "${SCHEDULE}"
      --save_dir "${VIDEO_DIR}"
      --split "${SPLIT}"
      --batch_size "${BATCH_SIZE}"
      --seed "${SEED_VALUES[@]}"
      "${common_limit[@]}"
    )
    if [[ -n "${CHECKPOINT_PATH:-}" ]]; then args+=(--checkpoint_path "${CHECKPOINT_PATH}"); fi
    if [[ -n "${CONFIG_PATH:-}" ]]; then args+=(--config_path "${CONFIG_PATH}"); fi
    if [[ -n "${BASE_MODEL_PATH:-}" ]]; then args+=(--base_model_path "${BASE_MODEL_PATH}"); fi
    if [[ "${OVERWRITE}" == 1 ]]; then args+=(--overwrite); fi
    srun --unbuffered "${PYTHON}" -m causal_prompt.cf_generate "${args[@]}"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}." >&2
    exit 2
    ;;
esac
