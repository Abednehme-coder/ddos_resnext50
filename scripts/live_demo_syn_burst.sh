#!/usr/bin/env bash
# Controlled SYN burst toward *your own* ECS to exercise live capture (demo only).
#
# Run from your laptop (or a second VM), NOT on the detector host. Needs root for raw sockets.
# Uses bounded rate + packet count (no --flood). Do not point at third-party IPs.
#
# Prerequisites (one of):
#   sudo apt install hping3    # preferred
#   sudo apt install nmap      # for nping fallback
#
# Usage:
#   TARGET_IP=213.250.144.74 LIVE_DEMO_SYN_CONFIRM=yes sudo -E bash scripts/live_demo_syn_burst.sh
#
# Optional env:
#   SYN_PORT=9999          # high port; avoid 22 if you need SSH during demo
#   PACKETS=5000           # hard cap (default 5000)
#   INTERVAL_USEC=4000     # hping3 -i u4000 → ~250 SYN/s (default 4000)
#
set -euo pipefail

TARGET_IP="${TARGET_IP:?Set TARGET_IP to your detector ECS public IPv4.}"
if [[ "${LIVE_DEMO_SYN_CONFIRM:-}" != "yes" ]]; then
  echo "Refusing: set LIVE_DEMO_SYN_CONFIRM=yes after you accept cloud ToS / demo risk." >&2
  exit 2
fi

SYN_PORT="${SYN_PORT:-9999}"
PACKETS="${PACKETS:-5000}"
INTERVAL_USEC="${INTERVAL_USEC:-4000}"

echo "=== Live demo SYN burst (self-test only) ===" >&2
echo "Target: ${TARGET_IP}:${SYN_PORT}  packets: ${PACKETS}  interval: u${INTERVAL_USEC} (~$((1000000 / INTERVAL_USEC)) pkt/s cap)" >&2
echo "Stop early: Ctrl+C" >&2
sleep 1

if command -v hping3 >/dev/null 2>&1; then
  exec hping3 -S -p "${SYN_PORT}" -c "${PACKETS}" -i "u${INTERVAL_USEC}" "${TARGET_IP}"
fi

if command -v nping >/dev/null 2>&1; then
  # nping: approximate rate; cap with -c
  RATE=$((1000000 / INTERVAL_USEC))
  exec nping --tcp -p "${SYN_PORT}" --flags SYN --rate "${RATE}" -c "${PACKETS}" "${TARGET_IP}"
fi

echo "Install hping3 or nmap (nping), then re-run with sudo." >&2
exit 1
