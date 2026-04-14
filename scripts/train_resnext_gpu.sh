#!/usr/bin/env bash
# Run ResNeXt training on GPU (MindSpore). Requires CUDA 11.6 + env from MINDSPORE_GPU_SETUP.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Match MINDSPORE_GPU_SETUP.md — without these, MindSpore may only register CPU.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv Python: ${VENV_PY}" >&2
  exit 1
fi

# Defaults: 0.3s window dataset + same output dir as focal v2 training (override OUTPUT_DIR / DATA_ROOT).
# For full focal + f1_ddos early stop, prefer scripts/train_resnext_per_second_0p3.sh
# Example override:
#   OUTPUT_DIR=.../model/my_run ./scripts/train_resnext_gpu.sh --batch-size 64 --epochs 40

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_per_second_window_0p3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/per_second_0p3_focal_v2}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
  --early-stop 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --lr-schedule cosine \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  "$@"
