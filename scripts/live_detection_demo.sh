#!/usr/bin/env bash
# Clear, simple live demo: SSH + tail + minimal one-line view (best for slides / judging).
# From repo root:
#   bash scripts/live_detection_demo.sh
#   ECS_HOST=ubuntu@x.x.x.x bash scripts/live_detection_demo.sh
#   TAIL_INITIAL=5 bash scripts/live_detection_demo.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECS_HOST="${ECS_HOST:-root@213.250.144.74}"
REMOTE_LOG="${REMOTE_LOG:-/var/log/synflood-detector/detections.log}"
TAIL_INITIAL="${TAIL_INITIAL:-6}"

exec ssh "${ECS_HOST}" "sudo tail -n '${TAIL_INITIAL}' -f '${REMOTE_LOG}'" \
  | python3 -u "${REPO}/scripts/detections_log_demo.py"
