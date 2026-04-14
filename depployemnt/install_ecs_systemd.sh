#!/usr/bin/env bash
# Install or refresh synflood-detector.service on the Huawei ECS from this repo’s canonical unit file.
# Run from repo root after editing depployemnt/systemd/synflood-detector.service
#
#   bash depployemnt/install_ecs_systemd.sh
#   ECS_HOST=ubuntu@213.250.144.74 bash depployemnt/install_ecs_systemd.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="${SCRIPT_DIR}/systemd/synflood-detector.service"
ECS_HOST="${ECS_HOST:-root@213.250.144.74}"
REMOTE_UNIT="/etc/systemd/system/synflood-detector.service"

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "Missing ${UNIT_SRC}" >&2
  exit 1
fi

echo "Installing ${UNIT_SRC} -> ${ECS_HOST}:${REMOTE_UNIT}"
# tcpdump drops to user tcpdump when writing -w files; dir must be writable by that user (not root-only).
ssh "${ECS_HOST}" "sudo mkdir -p /var/capture && sudo chown tcpdump:tcpdump /var/capture"
scp "${UNIT_SRC}" "${ECS_HOST}:/tmp/synflood-detector.service"
ssh "${ECS_HOST}" "sudo mv /tmp/synflood-detector.service '${REMOTE_UNIT}' && sudo systemctl daemon-reload && sudo systemctl restart synflood-detector && sudo systemctl --no-pager status synflood-detector"

echo "Done."
