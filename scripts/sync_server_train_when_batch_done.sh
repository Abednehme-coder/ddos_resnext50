#!/usr/bin/env bash
# Workflow (typical):
#   1) Per-second-window images already on server under REMOTE_TRAIN → pull to LOCAL_DEST (phase 1).
#   2) Then server regenerates with TIME_WINDOW_SECONDS=0.5 → pull again to LOCAL_DEST_PHASE2 (phase 2).
#
# Usage:
#   bash scripts/sync_server_train_when_batch_done.sh
#   # Generation already finished — skip the 45s “disk settle” delay:
#   POST_BATCH_SLEEP_SEC=0 bash scripts/sync_server_train_when_batch_done.sh
# Env:
#   SERVER              default: root@91.98.68.228
#   REMOTE_TRAIN        default: /root/datasets/CICDDoS2019/images/train/
#   LOCAL_DEST          Phase 1 rsync target (default: <repo>/dataset/images_per_second_window/train/)
#   POST_BATCH_SLEEP_SEC  Seconds to sleep after server batch is gone before rsync (default: 45; use 0 if already idle)
#   CHAIN_HALF_SECOND_REGEN  default: 1 — after phase 1 rsync, SSH clean+regen with 0.5s windows, then phase 2 rsync
#   LOCAL_DEST_PHASE2   default: <repo>/dataset/images_per_second_window_0p5/train/
#   REMOTE_LOG_0P5      default: /root/huawei/process_zips_0p5s.log (server log for second run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER="${SERVER:-root@91.98.68.228}"
REMOTE_TRAIN="${REMOTE_TRAIN:-/root/datasets/CICDDoS2019/images/train/}"
LOCAL_DEST="${LOCAL_DEST:-${REPO}/dataset/images_per_second_window/train}"
CHAIN_HALF_SECOND_REGEN="${CHAIN_HALF_SECOND_REGEN:-1}"
LOCAL_DEST_PHASE2="${LOCAL_DEST_PHASE2:-${REPO}/dataset/images_per_second_window_0p5/train}"
REMOTE_LOG_0P5="${REMOTE_LOG_0P5:-/root/huawei/process_zips_0p5s.log}"
POST_BATCH_SLEEP_SEC="${POST_BATCH_SLEEP_SEC:-45}"

LOG_DIR="${REPO}/dataset/images_per_second_window"
LOG="${LOG_DIR}/sync_from_server.log"
mkdir -p "${LOG_DIR}" "${LOCAL_DEST}"

LOCK="${LOG_DIR}/.sync-watch.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another sync watcher holds ${LOCK}; exit." >&2
  exit 1
fi

# Exit: 0 = batch running, 1 = batch not running, other = SSH / remote error (caller must abort).
remote_batch_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=25 "${SERVER}" \
    "pgrep -f 'scripts/process_zips_in_batches.sh' >/dev/null 2>&1"
}

wait_until_no_remote_batch() {
  local phase_label="${1:-batch}"
  while true; do
    set +e
    remote_batch_status
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      echo "$(date -Iseconds) ${phase_label} still running on server, sleep 120s"
      sleep 120
      continue
    fi
    if [[ "${rc}" -eq 1 ]]; then
      return 0
    fi
    echo "$(date -Iseconds) SSH/remote check failed (exit ${rc}); aborting so we do not rsync on a bad link." >&2
    exit "${rc}"
  done
}

start_server_regen_0p5s() {
  echo "$(date -Iseconds) SSH: start 0.5s time-window regeneration on server (log ${REMOTE_LOG_0P5})"
  # Path expands on the laptop; same string is valid on the server.
  ssh -o BatchMode=yes -o ConnectTimeout=30 "${SERVER}" \
    "cd /root/huawei && \
      export PYTHON=/root/huawei/.venv/bin/python && \
      export CLEAN_TRAIN_IMAGES_FIRST=1 && \
      export CONVERSION_MODE=per_second_window && \
      export TIME_WINDOW_SECONDS=0.5 && \
      export WINDOW_LABEL_LOGIC=and && \
      export CLASSIFY_METHOD=cic_schedule && \
      export SYN_THRESHOLD=100 && \
      export PCAP_SAMPLE_STRATEGY=even && \
      export MAX_IMAGES=200 && \
      nohup bash scripts/process_zips_in_batches.sh >> ${REMOTE_LOG_0P5} 2>&1 & \
      echo started_0p5s_batch"
}

{
  echo "=== $(date -Iseconds) phase 1: wait for current server batch, then rsync (1s or in-flight job) ==="
  echo "SERVER=${SERVER} REMOTE_TRAIN=${REMOTE_TRAIN}"
  echo "LOCAL_DEST=${LOCAL_DEST} CHAIN_HALF_SECOND_REGEN=${CHAIN_HALF_SECOND_REGEN} POST_BATCH_SLEEP_SEC=${POST_BATCH_SLEEP_SEC}"

  wait_until_no_remote_batch "phase1"
  echo "$(date -Iseconds) phase1: no batch process; waiting ${POST_BATCH_SLEEP_SEC}s for final disk writes"
  sleep "${POST_BATCH_SLEEP_SEC}"
  echo "$(date -Iseconds) phase 1 rsync..."
  mkdir -p "${LOCAL_DEST}/ddos" "${LOCAL_DEST}/normal"
  rsync -avz --progress "${SERVER}:${REMOTE_TRAIN}" "${LOCAL_DEST}/"
  echo "$(date -Iseconds) phase 1 rsync finished."
  echo "phase1 ddos:   $(find "${LOCAL_DEST}/ddos" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
  echo "phase1 normal: $(find "${LOCAL_DEST}/normal" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"

  if [[ "${CHAIN_HALF_SECOND_REGEN}" == "1" ]]; then
    echo ""
    echo "=== $(date -Iseconds) phase 2: server regen TIME_WINDOW_SECONDS=0.5, then rsync ==="
    start_server_regen_0p5s
    sleep 8
    wait_until_no_remote_batch "phase2(0.5s)"
    echo "$(date -Iseconds) 0.5s batch finished; waiting ${POST_BATCH_SLEEP_SEC}s"
    sleep "${POST_BATCH_SLEEP_SEC}"
    echo "$(date -Iseconds) phase 2 rsync -> ${LOCAL_DEST_PHASE2}"
    mkdir -p "${LOCAL_DEST_PHASE2}/ddos" "${LOCAL_DEST_PHASE2}/normal"
    rsync -avz --progress "${SERVER}:${REMOTE_TRAIN}" "${LOCAL_DEST_PHASE2}/"
    echo "$(date -Iseconds) phase 2 rsync finished."
    echo "phase2 ddos:   $(find "${LOCAL_DEST_PHASE2}/ddos" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
    echo "phase2 normal: $(find "${LOCAL_DEST_PHASE2}/normal" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)"
  else
    echo "$(date -Iseconds) CHAIN_HALF_SECOND_REGEN!=1, skipping 0.5s regen"
  fi

  echo "=== $(date -Iseconds) all done ==="
} >>"${LOG}" 2>&1

echo "Done. Log: ${LOG}"
