#!/usr/bin/env bash
set -euo pipefail

# Unified Wan2.2 training launcher.
#
# Usage:
#   bash scripts/train_wan22.sh
#
# Optional env overrides:
#   STAGE=scm|rcm
#   DATA_BACKEND=opens2v|webdataset
#   WAN22_DIFFUSERS_ROOT=./model/Wan2.2-T2V-A14B-Diffusers
#   WAN22_RCM_ROOT=./model/Wan2.2-T2V-A14B-Diffusers-rcm
#   OPENS2V_ROOT=./datasets/OpenS2V-5M
#   TAR_PATH_PATTERN=./assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar
#   VAE_PATH=./model/Wan2.2-T2V-A14B/Wan2.1_VAE.pth
#   T5_PATH=./model/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
#   NEG_EMB_PATH=./model/Wan2.2-T2V-A14B/umT5_wan_negative_emb.pt
#   NPROC_PER_NODE=8
#   MASTER_PORT=29666
#   BATCH_SIZE=1
#   NUM_WORKERS=1
#   TARGET_RESOLUTION=480p
#   TARGET_ASPECT_RATIO=16:9
#   CP_SIZE=8
#   FSDP_SHARD_SIZE=8
#   MAX_ITER=100000
#   TEACHER_INIT_STRATEGY=average|teacher_1|teacher_2
#   TEACHER_INIT_LOW_NOISE_WEIGHT=0.7
#   TEACHER_INIT_MODULE_AWARE=true|false
#   TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED=0.3
#   TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY=0.4
#   TEACHER_INIT_LOW_NOISE_WEIGHT_LATE=0.8
#   TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD=0.85
#   TEACHER_BOUNDARY_RATIO=0.875

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

export PYTHONPATH="${WORKDIR}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${WORKDIR}/outputs}"

STAGE="${STAGE:-scm}"
DATA_BACKEND="${DATA_BACKEND:-webdataset}"

WAN22_DIFFUSERS_ROOT="${WAN22_DIFFUSERS_ROOT:-./model/Wan2.2-T2V-A14B-Diffusers}"
WAN22_RCM_ROOT="${WAN22_RCM_ROOT:-./model/Wan2.2-T2V-A14B-Diffusers-rcm}"
OPENS2V_ROOT="${OPENS2V_ROOT:-./datasets/OpenS2V-5M}"
TAR_PATH_PATTERN="${TAR_PATH_PATTERN:-./assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar}"

VAE_PATH="${VAE_PATH:-./model/Wan2.2-T2V-A14B/Wan2.1_VAE.pth}"
T5_PATH="${T5_PATH:-./model/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth}"
NEG_EMB_PATH="${NEG_EMB_PATH:-./model/Wan2.2-T2V-A14B/umT5_wan_negative_emb.pt}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29666}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TARGET_RESOLUTION="${TARGET_RESOLUTION:-480p}"
TARGET_ASPECT_RATIO="${TARGET_ASPECT_RATIO:-16:9}"
CP_SIZE="${CP_SIZE:-8}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-8}"
MAX_ITER="${MAX_ITER:-100000}"
TEACHER_INIT_STRATEGY="${TEACHER_INIT_STRATEGY:-average}"
TEACHER_INIT_LOW_NOISE_WEIGHT="${TEACHER_INIT_LOW_NOISE_WEIGHT:-0.7}"
TEACHER_INIT_MODULE_AWARE="${TEACHER_INIT_MODULE_AWARE:-true}"
TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED="${TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED:-0.3}"
TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY="${TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY:-0.4}"
TEACHER_INIT_LOW_NOISE_WEIGHT_LATE="${TEACHER_INIT_LOW_NOISE_WEIGHT_LATE:-0.8}"
TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD="${TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD:-0.85}"
TEACHER_BOUNDARY_RATIO="${TEACHER_BOUNDARY_RATIO:-0.875}"

TEACHER_CKPT_1="${WAN22_RCM_ROOT}/Wan2.2-T2V-A14B-transformer-rcm.pth"
TEACHER_CKPT_2="${WAN22_RCM_ROOT}/Wan2.2-T2V-A14B-transformer_2-rcm.pth"

if [[ ! -f "${TEACHER_CKPT_1}" || ! -f "${TEACHER_CKPT_2}" ]]; then
  echo "[INFO] Converted Wan2.2 checkpoints not found. Converting from Diffusers..."
  python scripts/convert_wan22_diffusers_to_rcm.py \
    --model_root "${WAN22_DIFFUSERS_ROOT}" \
    --output_dir "${WAN22_RCM_ROOT}" \
    --strict
fi

if [[ "${DATA_BACKEND}" == "opens2v" ]]; then
  if [[ "${STAGE}" == "scm" ]]; then
    EXPERIMENT="wan2pt2_a14b_res480p_t2v_scm_opens2v"
  elif [[ "${STAGE}" == "rcm" ]]; then
    EXPERIMENT="wan2pt2_a14b_res480p_t2v_rcm_opens2v"
  else
    echo "[ERROR] Unsupported STAGE=${STAGE}, expected scm or rcm"
    exit 1
  fi
elif [[ "${DATA_BACKEND}" == "webdataset" ]]; then
  if [[ "${STAGE}" == "scm" ]]; then
    EXPERIMENT="wan2pt2_a14b_res480p_t2v_scm"
  elif [[ "${STAGE}" == "rcm" ]]; then
    EXPERIMENT="wan2pt2_a14b_res480p_t2v_rcm"
  else
    echo "[ERROR] Unsupported STAGE=${STAGE}, expected scm or rcm"
    exit 1
  fi
else
  echo "[ERROR] Unsupported DATA_BACKEND=${DATA_BACKEND}, expected opens2v or webdataset"
  exit 1
fi

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "[INFO] Launching Wan2.2 training"
echo "[INFO] EXPERIMENT=${EXPERIMENT}"
echo "[INFO] DATA_BACKEND=${DATA_BACKEND}"
echo "[INFO] WAN22_RCM_ROOT=${WAN22_RCM_ROOT}"
echo "[INFO] TEACHER_INIT_STRATEGY=${TEACHER_INIT_STRATEGY}"
echo "[INFO] TEACHER_INIT_LOW_NOISE_WEIGHT=${TEACHER_INIT_LOW_NOISE_WEIGHT}"
echo "[INFO] TEACHER_INIT_MODULE_AWARE=${TEACHER_INIT_MODULE_AWARE}"
echo "[INFO] TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED=${TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED}"
echo "[INFO] TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY=${TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY}"
echo "[INFO] TEACHER_INIT_LOW_NOISE_WEIGHT_LATE=${TEACHER_INIT_LOW_NOISE_WEIGHT_LATE}"
echo "[INFO] TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD=${TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD}"
echo "[INFO] TEACHER_BOUNDARY_RATIO=${TEACHER_BOUNDARY_RATIO}"

if [[ "${DATA_BACKEND}" == "opens2v" ]]; then
  echo "[INFO] OPENS2V_ROOT=${OPENS2V_ROOT}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    -m scripts.train --config=rcm/configs/registry_distill.py -- \
    experiment="${EXPERIMENT}" \
    trainer.max_iter="${MAX_ITER}" \
    dataloader_train.index_json_pattern="${OPENS2V_ROOT}/Jsons/total_part*.json" \
    dataloader_train.videos_root="${OPENS2V_ROOT}/Videos" \
    dataloader_train.batch_size="${BATCH_SIZE}" \
    dataloader_train.num_workers="${NUM_WORKERS}" \
    dataloader_train.target_resolution="${TARGET_RESOLUTION}" \
    dataloader_train.target_aspect_ratio="${TARGET_ASPECT_RATIO}" \
    model_parallel.context_parallel_size="${CP_SIZE}" \
    model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
    model.config.resolution="${TARGET_RESOLUTION}" \
    model.config.teacher_ckpt="${TEACHER_CKPT_1}" \
    model.config.teacher_ckpt_2="${TEACHER_CKPT_2}" \
    model.config.teacher_init_strategy="${TEACHER_INIT_STRATEGY}" \
    model.config.teacher_init_low_noise_weight="${TEACHER_INIT_LOW_NOISE_WEIGHT}" \
    model.config.teacher_init_module_aware="${TEACHER_INIT_MODULE_AWARE}" \
    model.config.teacher_init_low_noise_weight_embed="${TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED}" \
    model.config.teacher_init_low_noise_weight_early="${TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY}" \
    model.config.teacher_init_low_noise_weight_late="${TEACHER_INIT_LOW_NOISE_WEIGHT_LATE}" \
    model.config.teacher_init_low_noise_weight_head="${TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD}" \
    model.config.teacher_boundary_ratio="${TEACHER_BOUNDARY_RATIO}" \
    model.config.tokenizer.vae_pth="${VAE_PATH}" \
    model.config.text_encoder_path="${T5_PATH}" \
    model.config.neg_embed_path="${NEG_EMB_PATH}"
else
  echo "[INFO] TAR_PATH_PATTERN=${TAR_PATH_PATTERN}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    -m scripts.train --config=rcm/configs/registry_distill.py -- \
    experiment="${EXPERIMENT}" \
    trainer.max_iter="${MAX_ITER}" \
    dataloader_train.tar_path_pattern="${TAR_PATH_PATTERN}" \
    dataloader_train.batch_size="${BATCH_SIZE}" \
    dataloader_train.num_workers="${NUM_WORKERS}" \
    model_parallel.context_parallel_size="${CP_SIZE}" \
    model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
    model.config.resolution="${TARGET_RESOLUTION}" \
    model.config.teacher_ckpt="${TEACHER_CKPT_1}" \
    model.config.teacher_ckpt_2="${TEACHER_CKPT_2}" \
    model.config.teacher_init_strategy="${TEACHER_INIT_STRATEGY}" \
    model.config.teacher_init_low_noise_weight="${TEACHER_INIT_LOW_NOISE_WEIGHT}" \
    model.config.teacher_init_module_aware="${TEACHER_INIT_MODULE_AWARE}" \
    model.config.teacher_init_low_noise_weight_embed="${TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED}" \
    model.config.teacher_init_low_noise_weight_early="${TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY}" \
    model.config.teacher_init_low_noise_weight_late="${TEACHER_INIT_LOW_NOISE_WEIGHT_LATE}" \
    model.config.teacher_init_low_noise_weight_head="${TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD}" \
    model.config.teacher_boundary_ratio="${TEACHER_BOUNDARY_RATIO}" \
    model.config.tokenizer.vae_pth="${VAE_PATH}" \
    model.config.text_encoder_path="${T5_PATH}" \
    model.config.neg_embed_path="${NEG_EMB_PATH}"
fi
