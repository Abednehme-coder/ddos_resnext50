#!/usr/bin/env bash
# Copy only the Python files needed for live capture → packet-images → ResNeXt inference on ECS.
# Does not upload training notebooks, datasets, or the whole repo — use deploy_ckpt_to_ecs.sh for weights.
#
# Live pipeline imports:
#   run_pipeline.py
#   scripts/pcap_to_images.py  (PCAP read, one image per packet, same preprocessing path as offline eval)
#
# Time window length (e.g. align with your 0.3 s training windows) is set by systemd / CLI:
#   --window-sec  (and caps like --max-images-per-window), not by this script.
#
# Usage (from repo root):
#   bash depployemnt/sync_pipeline_to_ecs.sh
#
# Optional:
#   ECS_HOST=ubuntu@213.250.144.74
#   REMOTE_REPO=/opt/synflood-detector/repo
#   NO_RESTART=1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

ECS_HOST="${ECS_HOST:-root@213.250.144.74}"
REMOTE_REPO="${REMOTE_REPO:-/opt/synflood-detector/repo}"

# Paths relative to repo root — keep in sync with run_pipeline.py imports.
PIPELINE_FILES=(
  "run_pipeline.py"
  "scripts/pcap_to_images.py"
)

echo "Syncing pipeline runtime (${#PIPELINE_FILES[@]} files) -> ${ECS_HOST}:${REMOTE_REPO}/"

for f in "${PIPELINE_FILES[@]}"; do
  if [[ ! -f "${REPO}/${f}" ]]; then
    echo "Missing: ${REPO}/${f}" >&2
    exit 1
  fi
done

ssh -o BatchMode=yes "${ECS_HOST}" "mkdir -p '${REMOTE_REPO}/scripts'"

for f in "${PIPELINE_FILES[@]}"; do
  echo "  ${f}"
  rsync -avz "${REPO}/${f}" "${ECS_HOST}:${REMOTE_REPO}/${f}"
done

echo "Done."

if [[ "${NO_RESTART:-0}" != "1" ]]; then
  ssh "${ECS_HOST}" "sudo systemctl restart synflood-detector && sudo systemctl --no-pager status synflood-detector || true"
else
  echo "Skipped restart (NO_RESTART=1). On ECS: sudo systemctl restart synflood-detector"
fi

echo "Check CLI: ssh ${ECS_HOST} '/opt/synflood-detector/venv/bin/python ${REMOTE_REPO}/run_pipeline.py --help | head -25'"
