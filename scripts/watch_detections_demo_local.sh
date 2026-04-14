#!/usr/bin/env bash
# Run **on the ECS host** (no SSH): simple live demo lines. Requires sudo for the log path.
#   sudo bash /opt/synflood-detector/repo/scripts/watch_detections_demo_local.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG="${DETECTIONS_LOG:-/var/log/synflood-detector/detections.log}"
TAIL_INITIAL="${TAIL_INITIAL:-6}"

exec sudo tail -n "${TAIL_INITIAL}" -f "${LOG}" \
  | python3 -u "${REPO}/scripts/detections_log_demo.py"
