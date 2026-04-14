#!/usr/bin/env bash
# Run focal/imbalanced per-second training; if it exits successfully, power off the machine.
# Training failure (non-zero exit) does NOT shut down.
#
# Usage:
#   ./scripts/train_per_second_focal_then_poweroff.sh
#   DELAY_SEC=120 ./scripts/train_per_second_focal_then_poweroff.sh   # 2 min to cancel with Ctrl+C
#   ACTION=suspend ./scripts/train_per_second_focal_then_poweroff.sh  # sleep instead of power off
#
# Extra args are passed through to train_resnext_per_second_imbalanced.sh / train_resnext.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DELAY_SEC="${DELAY_SEC:-90}"
ACTION="${ACTION:-poweroff}"

run_training() {
  bash "${SCRIPT_DIR}/train_resnext_per_second_imbalanced.sh" "$@"
}

if ! run_training "$@"; then
  status=$?
  echo "Training failed (exit ${status}). Not shutting down." >&2
  exit "${status}"
fi

echo ""
echo "Training finished successfully at $(date -Iseconds)."
echo "System will ${ACTION} in ${DELAY_SEC}s — press Ctrl+C here to cancel."
sleep "${DELAY_SEC}"

case "${ACTION}" in
  poweroff|off|shutdown)
    systemctl poweroff
    ;;
  suspend|sleep)
    systemctl suspend
    ;;
  reboot)
    systemctl reboot
    ;;
  *)
    echo "Unknown ACTION=${ACTION} (use poweroff, suspend, or reboot)" >&2
    exit 1
    ;;
esac
