# DDoS ResNeXt50

MindSpore scripts to train, evaluate, export, and run inference with a ResNeXt50-32x4d model for classifying packet-derived images (e.g., DDoS vs. normal traffic). Utilities are included to convert PCAPs to images and batch process capture directories.

## Research Foundation

This project implements and extends the methodology from **"ResNet-Based Detection of SYN Flood DDoS Attacks"** (Bazzi et al., 2023, Beirut Arab University) using **MindSpore** and **MindCV** frameworks for the Huawei ICT Competition.

**Key Improvements:**
- **Architecture**: ResNeXt50-32x4d (enhanced over ResNet-50)
- **Framework**: MindSpore (Huawei native AI framework)
- **Deployment**: Multi-format export (MindIR, AIR, ONNX)
- **Performance**: Validates paper's 97.5% accuracy with modern implementation

📄 **See `PROJECT_RELATION_TO_PAPER.md`** for detailed analysis of how this implementation relates to the research paper.  
📋 **See `COMPETITION_SUBMISSION_SUMMARY.md`** for competition submission highlights.

## Repository Layout
- `scripts/train_resnext.py` — train and optionally evaluate on train/val/test splits.
- `scripts/eval_metrics.py` — compute precision/recall/F1 on a dataset split.
- `scripts/export_model.py` — export a trained checkpoint to MindIR/AIR/ONNX.
- `scripts/infer_resnext.py` — run inference recursively over an images folder.
- `scripts/pcap_to_images.py` — convert a PCAP into grayscale packet images.
- `scripts/convert_batch.sh` — batch PCAP conversion with simple DDoS/normal labeling.
- `models/resnext50_32x4d_export.mindir` — example exported MindIR (reference only).

## Requirements
- Python 3.x
- MindSpore (tested with 1.8) and `mindcv` for ResNeXt50
- NumPy, Pillow
- `tshark` (for `convert_batch.sh` SYN counting), optional if converting manually

> Scripts default to `--device-target GPU`; pass `--device-target CPU` to run on CPU.

## Data Preparation
Images are expected in ImageFolder layout with `train/`, `val/`, and `test/` splits:

```
/path/to/images_root/
  train/
    ddos/
    normal/
  val/
    ddos/
    normal/
  test/
    ddos/
    normal/
```

### Convert a Single PCAP
```bash
python scripts/pcap_to_images.py \
  --pcap /path/to/file.pcap \
  --out /path/to/images_root/ddos/file \
  --prefix file \
  --max-images 200 \
  --img-size 32 \
  --syn-only  # optional: keep only TCP SYN packets
```

### Batch Convert a PCAP Directory
`scripts/convert_batch.sh` labels captures as `ddos` when SYN counts exceed a threshold and deletes PCAPs after successful conversion. Environment variables:
- `PCAP_DIR` (default: `$HOME/ddos_data/pcaps`)
- `IMG_ROOT` (default: `$HOME/ddos_data/images`)
- `BATCH_COUNT` (pcaps per run, default 50)
- `MAX_IMAGES` (default 200)
- `SYN_THRESHOLD` (default 200)
- `PYTHON` (default `python3`)

Example:
```bash
PCAP_DIR=/data/pcaps IMG_ROOT=/data/images SYN_THRESHOLD=300 ./scripts/convert_batch.sh
```

## Training
```bash
python scripts/train_resnext.py \
  --data-root /path/to/images_root \
  --batch-size 16 \
  --epochs 8 \
  --lr 1e-3 \
  --num-classes 2 \
  --device-target GPU
```
Checkpoints save to `./models`. To resume or evaluate only, pass `--ckpt path/to.ckpt` and `--eval-only` if you want to skip training.

## Evaluation Metrics
Compute per-class precision/recall/F1 on a split (default `test`):
```bash
python scripts/eval_metrics.py \
  --data-root /path/to/images_root \
  --split test \
  --batch-size 32 \
  --ckpt models/resnext50-8_386.ckpt \
  --device-target GPU
```

## Model Export
Export a trained checkpoint to MindIR/AIR/ONNX:
```bash
python scripts/export_model.py \
  --ckpt models/resnext50-8_386.ckpt \
  --out models/resnext50_32x4d_export \
  --format MINDIR \
  --num-classes 2 \
  --device-target GPU
```

## Inference
Run inference on images (recursive) and print class probabilities:
```bash
python scripts/infer_resnext.py \
  --images-dir /path/to/images_root/test \
  --ckpt models/resnext50-8_386.ckpt \
  --class-names "ddos,normal" \
  --device-target GPU
```

---
Troubleshooting: ensure MindSpore and mindcv versions are compatible. Scripts include a fallback `SiLU` definition for MindSpore 1.8.
