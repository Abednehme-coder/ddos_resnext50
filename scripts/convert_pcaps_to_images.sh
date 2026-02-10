#!/usr/bin/env bash
# Classify each PCAP as ddos or normal (Python, early exit), then convert to 224x224 images.
# Usage: run from repo root (~/huawei). Set env vars or edit below.
#
# Env vars:
#   PCAP_DIR       Directory containing .pcap files (default: current dir)
#   IMG_ROOT       Image output root with train/ddos, train/normal (default: ~/datasets/CICDDoS2019/images)
#   SYN_THRESHOLD  Above this SYN count -> ddos (default: 100)
#   MAX_IMAGES     Max images per PCAP (default: 200)
#   MAX_FILES      Max number of PCAPs to process (default: no limit)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-$SCRIPT_DIR/..}"
REPO=$(cd "$REPO" && pwd)
PCAP_DIR="${PCAP_DIR:-.}"
IMG_ROOT="${IMG_ROOT:-$HOME/datasets/CICDDoS2019/images}"
SYN_THRESHOLD="${SYN_THRESHOLD:-100}"
MAX_IMAGES="${MAX_IMAGES:-200}"
MAX_FILES="${MAX_FILES:-999999}"

CLASSIFIER="$REPO/scripts/classify_pcap_syn.py"
CONVERTER="$REPO/scripts/pcap_to_images.py"
# Use python3 if available (server often has no 'python' symlink)
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)}"

mkdir -p "$IMG_ROOT/train/ddos" "$IMG_ROOT/train/normal"
cd "$PCAP_DIR" || exit 1

n=0
for pcap in *.pcap; do
  [ -f "$pcap" ] || continue
  [ $n -ge "$MAX_FILES" ] && break

  # 1) Classify (ddos vs normal) using Python - fast, early exit for DDoS
  class=$("$PYTHON" "$CLASSIFIER" "$(pwd)/$pcap" "$SYN_THRESHOLD" 2>/dev/null) || class=normal

  if [ "$class" = "ddos" ]; then
    syn_arg="--syn-only"
  else
    syn_arg=""
  fi

  # 2) Convert to images into the folder for that class
  "$PYTHON" "$CONVERTER" --pcap "$(pwd)/$pcap" --out "$IMG_ROOT/train/$class" --prefix "${pcap%.pcap}" --max-images "$MAX_IMAGES" $syn_arg

  echo "$pcap -> $class"
  n=$((n + 1))
done

echo "Processed $n files. Images: ddos=$(find "$IMG_ROOT/train/ddos" -name '*.png' 2>/dev/null | wc -l) normal=$(find "$IMG_ROOT/train/normal" -name '*.png' 2>/dev/null | wc -l)"
