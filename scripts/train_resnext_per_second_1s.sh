#!/usr/bin/env bash
# Train on dataset/images_per_second_window/train (1s-window PNGs) using val/test from dataset/images.
# Checkpoints go to model/per_second_1s/ so the original model/ run is untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/per_second_1s}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/dataset/images_per_second_window}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/dataset/images}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv Python: ${VENV_PY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${TRAIN_ROOT}" \
  --val-data-root "${EVAL_ROOT}" \
  --test-data-root "${EVAL_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
  --early-stop 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --lr-schedule cosine \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  "$@"
