#!/usr/bin/env bash
# Copy a trained MindSpore checkpoint to the ECS synflood-detector layout and optionally restart the service.
#
# Prerequisites: SSH key access to ECS, same paths as depployemnt/OPERATIONS.md
#
# Usage (from repo root):
#   bash depployemnt/deploy_ckpt_to_ecs.sh
#
# Default SSH target is the Huawei ECS instance (synflood-detector), NOT the dataset server.
# Override user/host if needed, e.g. export ECS_HOST=ubuntu@213.250.144.74
#
# Optional:
#   LOCAL_CKPT=...   # default: model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt
#   REMOTE_CKPT=...  # default: .../model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt
#   NO_RESTART=1     # only copy, do not systemctl restart
#   BACKUP=1         # ssh backup of previous ckpt on server before overwrite
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_CKPT="${LOCAL_CKPT:-${REPO}/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt}"
REMOTE_DIR="${REMOTE_DIR:-/opt/synflood-detector/repo/model}"
# Canonical path on server (matches run_pipeline.py default and local repo layout)
REMOTE_CKPT="${REMOTE_CKPT:-${REMOTE_DIR}/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt}"

# Huawei Cloud ECS (live pipeline). Dataset/zip host is a different machine — see depployemnt/OPERATIONS.md
ECS_HOST="${ECS_HOST:-root@213.250.144.74}"

if [[ ! -f "${LOCAL_CKPT}" ]]; then
  echo "Local checkpoint not found: ${LOCAL_CKPT}" >&2
  exit 1
fi

echo "Local:  ${LOCAL_CKPT}"
echo "Remote: ${ECS_HOST}:${REMOTE_CKPT}"

ssh -o BatchMode=yes "${ECS_HOST}" "mkdir -p '$(dirname "${REMOTE_CKPT}")'"

if [[ "${BACKUP:-1}" == "1" ]]; then
  ssh "${ECS_HOST}" "test -f '${REMOTE_CKPT}' && cp -a '${REMOTE_CKPT}' '${REMOTE_CKPT}.bak.'\$(date +%Y%m%d%H%M%S) || true"
fi

scp "${LOCAL_CKPT}" "${ECS_HOST}:${REMOTE_CKPT}"

echo "Uploaded."

if [[ "${NO_RESTART:-0}" != "1" ]]; then
  ssh "${ECS_HOST}" "sudo systemctl restart synflood-detector && sudo systemctl --no-pager status synflood-detector || true"
else
  echo "Skipped restart (NO_RESTART=1). On ECS run: sudo systemctl restart synflood-detector"
fi

echo "Done. Tail: ssh ${ECS_HOST} 'sudo journalctl -u synflood-detector -f'"
