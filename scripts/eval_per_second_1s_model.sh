#!/usr/bin/env bash
# Extra evaluation for the per-second-window trained model (checkpoints in model/per_second_1s/).
# Uses train/ under images_per_second_window for class indexing (same as training) and
# val/ + test/ under dataset/images (same as training).
#
# This is NOT the primary 0.3s focal-v2 checkpoint; eval scripts allow it only via:
#   HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1
set -euo pipefail

export HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CKPT="${CKPT:-${ROOT}/model/per_second_1s/resnext50_32x4d_best.ckpt}"
VENV_PY="${ROOT}/.venv/bin/python"
EVAL_PY="${ROOT}/scripts/eval_test_detailed.py"
BLIND_PY="${ROOT}/scripts/blind_test_random.py"

# Match scripts/train_resnext_gpu.sh so MindSpore finds CUDA/cuDNN when using GPU.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DEVICE_TARGET="${DEVICE_TARGET:-GPU}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "========== Detailed eval: val (confusion matrix, P/R/F1) =========="
"${VENV_PY}" "${EVAL_PY}" \
  --ckpt "${CKPT}" \
  --data-root "${ROOT}/dataset/images_per_second_window" \
  --eval-root "${ROOT}/dataset/images" \
  --split val \
  --device-target "${DEVICE_TARGET}"

echo ""
echo "========== Detailed eval: test =========="
"${VENV_PY}" "${EVAL_PY}" \
  --ckpt "${CKPT}" \
  --data-root "${ROOT}/dataset/images_per_second_window" \
  --eval-root "${ROOT}/dataset/images" \
  --split test \
  --device-target "${DEVICE_TARGET}"

echo ""
echo "========== Blind test: random sample from test/ (balanced per class) =========="
BLIND_TOTAL="${BLIND_TOTAL:-2000}"
"${VENV_PY}" "${BLIND_PY}" \
  --ckpt "${CKPT}" \
  --data-root "${ROOT}/dataset/images" \
  --split test \
  --total "${BLIND_TOTAL}" \
  --device-target "${DEVICE_TARGET}" \
  --seed 42

echo ""
echo "Done. DEVICE_TARGET=${DEVICE_TARGET} (set DEVICE_TARGET=CPU if GPU eval fails)."
echo "Larger blind sample: BLIND_TOTAL=5000 ./scripts/eval_per_second_1s_model.sh"
