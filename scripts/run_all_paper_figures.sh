#!/usr/bin/env bash
# Generate PDF figures for the research report (research/figures/). Defaults to GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${ROOT}/research/figures"
exec "${PY}" "${ROOT}/scripts/eval_roc_pr_calibration.py" \
  --ckpt "${ROOT}/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt" \
  --data-root "${ROOT}/dataset/images_per_second_window_0p3" \
  --split test \
  --out-dir "${ROOT}/research/figures" \
  --device-target "${DEVICE_TARGET}" \
  "$@"
