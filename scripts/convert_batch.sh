#!/usr/bin/env bash
# Convert a batch of pcaps to images and delete pcaps on successful conversion.
# Configurable via env vars:
#   PCAP_DIR       (default: $HOME/ddos_data/pcaps)
#   IMG_ROOT       (default: $HOME/ddos_data/images)
#   BATCH_COUNT    (default: 50)   # how many pcaps to process per run
#   MAX_IMAGES     (default: 200)  # max images per pcap
#   SYN_THRESHOLD  (default: 200)  # SYN count above which class is ddos
#   PYTHON         (default: python3)

set -uo pipefail

PCAP_DIR="${PCAP_DIR:-$HOME/ddos_data/pcaps}"
IMG_ROOT="${IMG_ROOT:-$HOME/ddos_data/images}"
BATCH_COUNT="${BATCH_COUNT:-50}"
MAX_IMAGES="${MAX_IMAGES:-200}"
SYN_THRESHOLD="${SYN_THRESHOLD:-200}"
PYTHON="${PYTHON:-python3}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONVERTER="$SCRIPT_DIR/pcap_to_images.py"

if [ ! -x "$CONVERTER" ] && [ ! -f "$CONVERTER" ]; then
  echo "Converter not found at $CONVERTER" >&2
  exit 1
fi

if [ ! -d "$PCAP_DIR" ]; then
  echo "PCAP_DIR does not exist: $PCAP_DIR" >&2
  exit 1
fi

mapfile -t pcaps < <(ls "$PCAP_DIR" | head -n "$BATCH_COUNT")
if [ "${#pcaps[@]}" -eq 0 ]; then
  echo "No pcaps to process in $PCAP_DIR"
  exit 0
fi

total=${#pcaps[@]}
success=0
failed=0
kept=0

echo "Processing $total pcaps from $PCAP_DIR (max $BATCH_COUNT)."
echo "Images root: $IMG_ROOT | Max images per pcap: $MAX_IMAGES | SYN threshold: $SYN_THRESHOLD"

for pcap in "${pcaps[@]}"; do
  pcap_path="$PCAP_DIR/$pcap"
  if [ ! -f "$pcap_path" ]; then
    echo "Skip missing: $pcap"
    ((failed++))
    continue
  fi

  syn_count=$(tshark -r "$pcap_path" -Y "tcp.flags.syn==1" -T fields -e frame.number | wc -l)
  class=normal
  syn_arg=()
  if [ "$syn_count" -gt "$SYN_THRESHOLD" ]; then
    class=ddos
    syn_arg=(--syn-only)
  fi

  out_dir="$IMG_ROOT/$class/$pcap"
  mkdir -p "$out_dir"

  args=(--pcap "$pcap_path" --out "$out_dir" --prefix "$pcap" --max-images "$MAX_IMAGES")
  args+=("${syn_arg[@]}")

  echo "[$((success+failed+kept+1))/$total] $pcap -> $class (SYNs=$syn_count)"
  if "$PYTHON" "$CONVERTER" "${args[@]}"; then
    img_count=$(find "$out_dir" -maxdepth 1 -type f -name '*.png' | wc -l)
    if [ "$img_count" -gt 0 ]; then
      rm -f "$pcap_path"
      ((success++))
    else
      echo "No images written; keeping $pcap"
      ((kept++))
    fi
  else
    echo "Conversion failed; keeping $pcap"
    ((failed++))
  fi
done

echo "Done. Success: $success | Failed: $failed | Kept (no images): $kept"
