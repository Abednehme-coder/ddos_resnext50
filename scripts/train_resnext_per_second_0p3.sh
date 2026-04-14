#!/usr/bin/env bash
# Train on 0.3s time-window PNGs. Val/test MUST come from the same distribution as train
# (run scripts/split_train_val_test.py --images-root dataset/images_per_second_window_0p3 first).
# Same imbalance-aware defaults: focal, f1_ddos early stop.
# Checkpoints: model/per_second_0p3_focal/ (override with OUTPUT_DIR=...)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"
# Default output dir v2: trained with val/test from same 0.3s split (not dataset/images).
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/per_second_0p3_focal_v2}"
# Single root: train/ val/ test/ are all 0.3s window images (aligned distribution).
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_per_second_window_0p3}"

mkdir -p "${OUTPUT_DIR}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${DATA_ROOT}" \
  --val-data-root "${DATA_ROOT}" \
  --test-data-root "${DATA_ROOT}" \
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
