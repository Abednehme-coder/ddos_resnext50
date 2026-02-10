#!/usr/bin/env bash
# Pick N random images from train/ddos and train/normal, copy to a folder
# you can then pull via scp from your Windows machine.
#
# Usage:
#   ./scripts/pick_random_images_for_transfer.sh [N]
#   N = number per class (default 5), so 2*N images total.
#
# Then from Windows (CMD, not SSH):
#   scp -r user@server:~/preview_transfer C:\Users\user\Desktop\preview_transfer

set -e
N="${1:-5}"
IMG_ROOT="${IMG_ROOT:-$HOME/datasets/CICDDoS2019/images}"
OUT_DIR="${OUT_DIR:-$HOME/preview_transfer}"

TRAIN_DDOS="$IMG_ROOT/train/ddos"
TRAIN_NORMAL="$IMG_ROOT/train/normal"

if [[ ! -d "$TRAIN_DDOS" || ! -d "$TRAIN_NORMAL" ]]; then
  echo "Error: train dirs not found at $IMG_ROOT/train/{ddos,normal}" >&2
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/ddos" "$OUT_DIR/normal"

# Use shuf to pick N random files per class (fallback: sort -R)
pick_random() {
  local dir="$1"
  local n="$2"
  if command -v shuf &>/dev/null; then
    ls "$dir" | shuf -n "$n"
  else
    ls "$dir" | sort -R | head -n "$n"
  fi
}

count_ddos=$(ls "$TRAIN_DDOS" 2>/dev/null | wc -l)
count_normal=$(ls "$TRAIN_NORMAL" 2>/dev/null | wc -l)
if [[ "$count_ddos" -lt "$N" ]]; then N_ddos=$count_ddos; else N_ddos=$N; fi
if [[ "$count_normal" -lt "$N" ]]; then N_normal=$count_normal; else N_normal=$N; fi

pick_random "$TRAIN_DDOS" "$N_ddos" | while read -r f; do
  cp "$TRAIN_DDOS/$f" "$OUT_DIR/ddos/"
done
pick_random "$TRAIN_NORMAL" "$N_normal" | while read -r f; do
  cp "$TRAIN_NORMAL/$f" "$OUT_DIR/normal/"
done

echo "Copied $N_ddos ddos + $N_normal normal images to $OUT_DIR"
echo ""
echo "To download to your Windows machine, run this from CMD or PowerShell (not SSH):"
echo "  scp -r $(whoami)@<YOUR_SERVER_IP>:$OUT_DIR %USERPROFILE%\\Desktop\\preview_transfer"
echo ""
echo "Example if server is 91.98.68.228 and user is abed:"
echo "  scp -r abed@91.98.68.228:$OUT_DIR %USERPROFILE%\\Desktop\\preview_transfer"
