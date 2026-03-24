#!/usr/bin/env bash
set -euo pipefail

# Very-short-run ablation launcher for Wan2.2 A14B prototype.
#
# This script helps compare initialization/routing defaults under the same
# short training budget before committing to large-scale runs.
#
# Usage:
#   bash scripts/run_wan22_short_ablation.sh
#
# Optional env overrides:
#   STAGE=scm|rcm
#   DATA_BACKEND=webdataset|opens2v
#   MAX_ITER=300
#   ABLATION_OUTPUT_ROOT=./outputs/wan22_short_ablation
#   ABLATION_DRY_RUN=true|false

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

STAGE="${STAGE:-scm}"
DATA_BACKEND="${DATA_BACKEND:-webdataset}"
MAX_ITER="${MAX_ITER:-300}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${WORKDIR}/outputs/wan22_short_ablation}"
ABLATION_DRY_RUN="${ABLATION_DRY_RUN:-false}"

mkdir -p "${ABLATION_OUTPUT_ROOT}"

# case_name|init_strategy|module_aware|global_w|embed_w|early_w|late_w|head_w|boundary
CASES=(
  "avg_plain_r080|average|false|0.5|0.5|0.5|0.5|0.5|0.8"
  "avg_plain_r0875|average|false|0.5|0.5|0.5|0.5|0.5|0.875"
  "avg_plain_r090|average|false|0.5|0.5|0.5|0.5|0.5|0.9"
  "avg_module_r0875|average|true|0.5|0.3|0.4|0.8|0.85|0.875"
)

echo "[INFO] Wan2.2 short ablation"
echo "[INFO] STAGE=${STAGE}"
echo "[INFO] DATA_BACKEND=${DATA_BACKEND}"
echo "[INFO] MAX_ITER=${MAX_ITER}"
echo "[INFO] OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT}"

for row in "${CASES[@]}"; do
  IFS='|' read -r CASE_NAME INIT_STRATEGY MODULE_AWARE GLOBAL_W EMBED_W EARLY_W LATE_W HEAD_W BOUNDARY <<<"${row}"
  CASE_OUT="${ABLATION_OUTPUT_ROOT}/${CASE_NAME}"
  mkdir -p "${CASE_OUT}"

  CMD=(
    env
    STAGE="${STAGE}"
    DATA_BACKEND="${DATA_BACKEND}"
    MAX_ITER="${MAX_ITER}"
    IMAGINAIRE_OUTPUT_ROOT="${CASE_OUT}"
    TEACHER_INIT_STRATEGY="${INIT_STRATEGY}"
    TEACHER_INIT_MODULE_AWARE="${MODULE_AWARE}"
    TEACHER_INIT_LOW_NOISE_WEIGHT="${GLOBAL_W}"
    TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED="${EMBED_W}"
    TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY="${EARLY_W}"
    TEACHER_INIT_LOW_NOISE_WEIGHT_LATE="${LATE_W}"
    TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD="${HEAD_W}"
    TEACHER_BOUNDARY_RATIO="${BOUNDARY}"
    WAN22_PREFLIGHT=true
    WAN22_PREFLIGHT_PARITY=false
    WAN22_PREFLIGHT_ROUTING=true
    WAN22_PREFLIGHT_TRAINING_STATE=true
    WAN22_PREFLIGHT_REPORT_DIR="${CASE_OUT}/preflight"
    bash scripts/train_wan22.sh
  )

  echo "[INFO] Running case ${CASE_NAME}"
  if [[ "${ABLATION_DRY_RUN}" == "true" ]]; then
    printf '  %q' "${CMD[@]}"
    printf '\n'
  else
    "${CMD[@]}"
  fi
done

echo "[done] Wan2.2 short ablation finished"
