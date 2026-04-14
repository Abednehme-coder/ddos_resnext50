#!/usr/bin/env bash
# Run all standard offline evaluations for model/per_second_0p3_focal_v2 (0.3 s windows).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CKPT="${CKPT:-${ROOT}/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt}"
DATA="${DATA:-${ROOT}/dataset/images_per_second_window_0p3}"
VENV_PY="${ROOT}/.venv/bin/python"
# Default to GPU for local evaluation; use DEVICE=CPU if no CUDA.
DEVICE="${DEVICE:-GPU}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA}/train" ]]; then
  echo "Missing dataset: ${DATA}" >&2
  exit 1
fi
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv Python: ${VENV_PY}" >&2
  exit 1
fi

echo "========== val (0.3s same-domain) =========="
"${VENV_PY}" "${ROOT}/scripts/eval_test_detailed.py" \
  --ckpt "${CKPT}" --data-root "${DATA}" --split val --device-target "${DEVICE}" "$@"

echo ""
echo "========== test (0.3s same-domain) =========="
"${VENV_PY}" "${ROOT}/scripts/eval_test_detailed.py" \
  --ckpt "${CKPT}" --data-root "${DATA}" --split test --device-target "${DEVICE}" "$@"

# Max equal blind sample size = 2 * min(|ddos|, |normal|) on test; typically 270 for this split.
echo ""
echo "========== blind test (test, 270 total, seed 42) =========="
"${VENV_PY}" "${ROOT}/scripts/blind_test_random.py" \
  --data-root "${DATA}" --ckpt "${CKPT}" --split test --total 270 --seed 42 \
  --device-target "${DEVICE}"

echo ""
echo "Done. Full log template: reports/eval_results_per_second_0p3.md"
