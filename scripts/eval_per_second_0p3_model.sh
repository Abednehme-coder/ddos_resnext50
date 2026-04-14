#!/usr/bin/env bash
# Eval 0.3s checkpoint on the SAME domain as training (images_per_second_window_0p3 train/val/test).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CKPT="${CKPT:-${ROOT}/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt}"
VENV_PY="${ROOT}/.venv/bin/python"
EVAL_PY="${ROOT}/scripts/eval_test_detailed.py"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/dataset/images_per_second_window_0p3}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "========== val (same-domain 0.3s) =========="
"${VENV_PY}" "${EVAL_PY}" --ckpt "${CKPT}" --data-root "${DATA_ROOT}" --split val --device-target "${DEVICE_TARGET}"

echo ""
echo "========== test (same-domain 0.3s) =========="
"${VENV_PY}" "${EVAL_PY}" --ckpt "${CKPT}" --data-root "${DATA_ROOT}" --split test --device-target "${DEVICE_TARGET}"
