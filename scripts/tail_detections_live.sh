#!/usr/bin/env bash
# Live, human-readable detection lines (SSH + unbuffered pretty-print — more columns than demo).
# For a minimal one-line demo view use: bash scripts/live_detection_demo.sh
# Usage from repo root:
#   bash scripts/tail_detections_live.sh
#   ECS_HOST=ubuntu@x.x.x.x bash scripts/tail_detections_live.sh
#   TAIL_INITIAL=15 bash scripts/tail_detections_live.sh   # more context lines before follow
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECS_HOST="${ECS_HOST:-root@213.250.144.74}"
REMOTE_LOG="${REMOTE_LOG:-/var/log/synflood-detector/detections.log}"
TAIL_INITIAL="${TAIL_INITIAL:-8}"

exec ssh "${ECS_HOST}" "sudo tail -n '${TAIL_INITIAL}' -f '${REMOTE_LOG}'" \
  | python3 -u "${REPO}/scripts/detections_log_pretty.py"
