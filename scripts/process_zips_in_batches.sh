#!/usr/bin/env bash
# Process CICDDoS2019 zip files one at a time: unzip -> rename extensionless to .pcap -> classify + generate images -> delete unzipped.
# Keeps images in one place (IMG_ROOT/train/ddos, train/normal). Run from repo root on the server.
#
# Usage:
#   cd ~/huawei && bash scripts/process_zips_in_batches.sh
#
# Env (optional):
#   PCAP_ZIP_DIR   Directory containing the .zip files (default: ~/datasets/CICDDoS2019/PCAPs)
#   IMG_ROOT       Output images here (default: ~/datasets/CICDDoS2019/images)
#   STAGING_DIR    Where to unzip (default: $PCAP_ZIP_DIR/staging)
#   REPO           Repo root (default: script's parent dir)
#   CLASSIFY_METHOD   cic_schedule (default for this script) or syn_count — see convert_pcaps_to_images.sh
#   CIC_SCHEDULE_CONFIG  Path to JSON schedule (default: $REPO/scripts/cic_ddos2019_syn_windows.json)
#   CLEAN_TRAIN_IMAGES_FIRST  If set to 1, delete all *.png under train/ddos and train/normal before processing (fresh labels)
#   PYTHON            e.g. $REPO/.venv/bin/python on servers where system python lacks deps
#   PCAP_SAMPLE_STRATEGY  even (default) or sequential — see convert_pcaps_to_images.sh
#   SYN_ONLY_DDOS     1 (default): ddos images from SYN packets only; 0: all packets for ddos too
#   PACKETS_PER_IMAGE 10 (default): N eligible packets per image; set 1 for single-packet datasets
#   CONVERSION_MODE   per_pcap (default) or per_second_window — see convert_pcaps_to_images.sh
#   TIME_WINDOW_SECONDS  1 (default) for per_second_window
#   WINDOW_LABEL_LOGIC   and (default) or or

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-$SCRIPT_DIR/..}"
REPO=$(cd "$REPO" && pwd)
CLASSIFY_METHOD="${CLASSIFY_METHOD:-cic_schedule}"
CIC_SCHEDULE_CONFIG="${CIC_SCHEDULE_CONFIG:-$REPO/scripts/cic_ddos2019_syn_windows.json}"
CLEAN_TRAIN_IMAGES_FIRST="${CLEAN_TRAIN_IMAGES_FIRST:-0}"
PCAP_SAMPLE_STRATEGY="${PCAP_SAMPLE_STRATEGY:-even}"
SYN_ONLY_DDOS="${SYN_ONLY_DDOS:-1}"
PACKETS_PER_IMAGE="${PACKETS_PER_IMAGE:-10}"
CONVERSION_MODE="${CONVERSION_MODE:-per_pcap}"
TIME_WINDOW_SECONDS="${TIME_WINDOW_SECONDS:-1}"
WINDOW_LABEL_LOGIC="${WINDOW_LABEL_LOGIC:-and}"
export CLASSIFY_METHOD CIC_SCHEDULE_CONFIG PYTHON PCAP_SAMPLE_STRATEGY SYN_ONLY_DDOS PACKETS_PER_IMAGE
export CONVERSION_MODE TIME_WINDOW_SECONDS WINDOW_LABEL_LOGIC
PCAP_ZIP_DIR="${PCAP_ZIP_DIR:-$HOME/datasets/CICDDoS2019/PCAPs}"
IMG_ROOT="${IMG_ROOT:-$HOME/datasets/CICDDoS2019/images}"
STAGING_DIR="${STAGING_DIR:-$PCAP_ZIP_DIR/staging}"

PCAP_ZIP_DIR=$(cd "$PCAP_ZIP_DIR" && pwd)
mkdir -p "$STAGING_DIR"
STAGING_DIR=$(cd "$STAGING_DIR" && pwd)

# All zips, one at a time
ZIPS=(
  "PCAP-01-12_0-0249.zip"
  "PCAP-01-12_0250-0499.zip"
  "PCAP-01-12_0500-0749.zip"
  "PCAP-01-12_0750-0818.zip"
  "PCAP-03-11.zip"
)

# CICDDoS2019 zips contain PCAP files with NO extension (e.g. SAT-01-12-2018_0750). Rename to .pcap so find and convert script see them.
rename_to_pcap() {
  local root="$1"
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    [ "${f%.pcap}" = "$f" ] || continue
    mv -- "$f" "$f.pcap"
  done < <(find "$root" -type f)
}

# Run conversion for every directory that contains .pcap files
run_convert_for_all_pcap_dirs() {
  local root="$1"
  local dirs
  dirs=$(find "$root" -type f -name "*.pcap" -printf "%h\n" | sort -u)
  if [ -z "$dirs" ]; then
    echo "No .pcap files found under $root"
    return 0
  fi
  while IFS= read -r dir; do
    [ -d "$dir" ] || continue
    if [ -n "$(find "$dir" -maxdepth 1 -name '*.pcap' -print -quit 2>/dev/null)" ]; then
      echo "=== Converting PCAPs in: $dir ==="
      export IMG_ROOT
      export PCAP_DIR="$dir"
      bash "$REPO/scripts/convert_pcaps_to_images.sh"
    fi
  done <<< "$dirs"
}

process_one_zip() {
  local z="$1"
  local path="$PCAP_ZIP_DIR/$z"
  if [ ! -f "$path" ]; then
    echo "Skip (not found): $path"
    return 0
  fi
  echo ""
  echo "========== ZIP: $z =========="
  rm -rf "${STAGING_DIR:?}"/*
  echo "Unzipping..."
  unzip -o -q "$path" -d "$STAGING_DIR"
  echo "Renaming extensionless files to .pcap..."
  rename_to_pcap "$STAGING_DIR"
  run_convert_for_all_pcap_dirs "$STAGING_DIR"
  echo "Clearing staging..."
  rm -rf "${STAGING_DIR:?}"/*
  echo "Done: $z"
}

clean_train_image_dirs() {
  if [[ "$CLEAN_TRAIN_IMAGES_FIRST" != "1" ]]; then
    return 0
  fi
  echo ""
  echo "=== CLEAN_TRAIN_IMAGES_FIRST=1: removing old PNGs under $IMG_ROOT/train/ddos and .../normal ==="
  mkdir -p "$IMG_ROOT/train/ddos" "$IMG_ROOT/train/normal"
  find "$IMG_ROOT/train/ddos" -maxdepth 1 -type f -name '*.png' -delete
  find "$IMG_ROOT/train/normal" -maxdepth 1 -type f -name '*.png' -delete
  echo "Clean done."
}

main() {
  echo "Repo: $REPO"
  echo "PCAP zips dir: $PCAP_ZIP_DIR"
  echo "Image root: $IMG_ROOT"
  echo "Staging: $STAGING_DIR"
  echo "CLASSIFY_METHOD=$CLASSIFY_METHOD CIC_SCHEDULE_CONFIG=$CIC_SCHEDULE_CONFIG CLEAN_TRAIN_IMAGES_FIRST=$CLEAN_TRAIN_IMAGES_FIRST PCAP_SAMPLE_STRATEGY=$PCAP_SAMPLE_STRATEGY SYN_ONLY_DDOS=$SYN_ONLY_DDOS PACKETS_PER_IMAGE=$PACKETS_PER_IMAGE CONVERSION_MODE=$CONVERSION_MODE TIME_WINDOW_SECONDS=$TIME_WINDOW_SECONDS WINDOW_LABEL_LOGIC=$WINDOW_LABEL_LOGIC"
  clean_train_image_dirs
  for z in "${ZIPS[@]}"; do
    process_one_zip "$z"
  done
  echo ""
  echo "All zips done. Images: ddos=$(find "$IMG_ROOT/train/ddos" -name '*.png' 2>/dev/null | wc -l) normal=$(find "$IMG_ROOT/train/normal" -name '*.png' 2>/dev/null | wc -l)"
}

main
