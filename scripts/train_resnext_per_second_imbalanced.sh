#!/usr/bin/env bash
# Per-second-window train with imbalance-aware defaults: focal loss, minority boost,
# early stopping on val DDoS F1 (f1_ddos). Checkpoints: model/per_second_1s_focal/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/per_second_1s_focal}"

mkdir -p "${OUTPUT_DIR}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${PROJECT_ROOT}/dataset/images_per_second_window" \
  --val-data-root "${PROJECT_ROOT}/dataset/images_per_second_window" \
  --test-data-root "${PROJECT_ROOT}/dataset/images_per_second_window" \
  --output-dir "${OUTPUT_DIR}" \
  --loss focal \
  --focal-gamma 2.0 \
  --minority-boost 8.0 \
  --early-stop-metric f1_ddos \
  --early-stop 10 \
  --epochs 40 \
  --batch-size 32 \
  --lr 5e-4 \
  --lr-schedule cosine \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  "$@"
