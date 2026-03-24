#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${WORKDIR}"

export PYTHONPATH="${WORKDIR}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${WORKDIR}/outputs}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${WORKDIR}/model/Wan2.1-T2V-distill}"
DATASET_ROOT="${DATASET_ROOT:-${WORKDIR}/assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K}"

REGISTRY="${REGISTRY:-registry_distill}"
EXPERIMENT="${EXPERIMENT:-wan2pt1_1pt3B_res480p_t2v_rCM}"
TEACHER_CKPT="${TEACHER_CKPT:-${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp}"
VAE_PATH="${VAE_PATH:-${CHECKPOINT_ROOT}/Wan2.1_VAE.pth}"
T5_PATH="${T5_PATH:-${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
NEG_EMB_PATH="${NEG_EMB_PATH:-${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29666}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CP_SIZE="${CP_SIZE:-8}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-8}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"

# Visualization callback is very memory hungry; keep it off by default on multi-GPU training.
ENABLE_VIZ_SAMPLE="${ENABLE_VIZ_SAMPLE:-false}"
VIZ_RUN_AT_START="${VIZ_RUN_AT_START:-false}"
VIZ_SAMPLE_FIX="${VIZ_SAMPLE_FIX:-false}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-1}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "${WANDB_API_KEY}"
fi

if (( NPROC_PER_NODE % CP_SIZE != 0 )); then
    echo "[ERROR] NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by CP_SIZE (${CP_SIZE})."
    exit 1
fi

echo "[INFO] Launching training"
echo "[INFO] EXPERIMENT=${EXPERIMENT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}, CP_SIZE=${CP_SIZE}, FSDP_SHARD_SIZE=${FSDP_SHARD_SIZE}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    -m scripts.train --config="rcm/configs/${REGISTRY}.py" -- \
    experiment="${EXPERIMENT}" \
    model.config.teacher_ckpt="${TEACHER_CKPT}" \
    model.config.tokenizer.vae_pth="${VAE_PATH}" \
    model.config.text_encoder_path="${T5_PATH}" \
    model.config.neg_embed_path="${NEG_EMB_PATH}" \
    dataloader_train.tar_path_pattern="${DATASET_ROOT}/shard*.tar" \
    dataloader_train.batch_size=${BATCH_SIZE} \
    dataloader_train.num_workers=${NUM_WORKERS} \
    model_parallel.context_parallel_size=${CP_SIZE} \
    model.config.fsdp_shard_size=${FSDP_SHARD_SIZE} \
    trainer.grad_accum_iter=${GRAD_ACCUM_ITER} \
    trainer.callbacks.every_n_sample_reg.is_sample=${ENABLE_VIZ_SAMPLE} \
    trainer.callbacks.every_n_sample_ema.is_sample=${ENABLE_VIZ_SAMPLE} \
    trainer.callbacks.every_n_sample_reg.run_at_start=${VIZ_RUN_AT_START} \
    trainer.callbacks.every_n_sample_ema.run_at_start=${VIZ_RUN_AT_START} \
    trainer.callbacks.every_n_sample_reg.sample_fix=${VIZ_SAMPLE_FIX} \
    trainer.callbacks.every_n_sample_ema.sample_fix=${VIZ_SAMPLE_FIX} \
    trainer.callbacks.every_n_sample_reg.num_samples=${VIZ_NUM_SAMPLES} \
    trainer.callbacks.every_n_sample_ema.num_samples=${VIZ_NUM_SAMPLES}
