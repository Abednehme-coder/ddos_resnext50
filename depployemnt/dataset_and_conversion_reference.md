## Dataset formation & PCAP -> image conversion (authoritative)

This document records the exact logic used in this repo so real-time conversion can match training.

### 1) PCAP -> label for training (offline only)
File: `scripts/classify_pcap_syn.py`

Two modes (`--method` or env `CLASSIFY_METHOD`):

**A) `syn_count` (default)**  
- Reads packets using `iter_pcap_packets()` (lightweight pcap parsing; no scapy).
- Counts TCP SYN packets using `is_tcp_syn(packet_bytes)`.
- Early exits once `count > threshold`:
  - returns `"ddos"` if above threshold
  - returns `"normal"` otherwise

Key implementation:
- `is_tcp_syn()` checks:
  - Ethernet type is IPv4 (`0x0800`), handles optional 802.1Q VLAN tagging
  - IP protocol is TCP (`proto == 6`)
  - reads TCP flags and checks SYN bit: `(flags & 0x02) != 0`

**B) `cic_schedule`** (optional)  
- Uses `scripts/pcap_time.py` → `read_pcap_time_span_utc()` for first/last packet time in the PCAP (UTC).
- Loads SYN flood time windows from `scripts/cic_ddos2019_syn_windows.json` (cite [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) / Sharafaldin et al., 2019).
- Maps capture **calendar day** using filename `DD-MM-YYYY` (e.g. `SAT-03-11-2018_…`) or first packet local date, via `dd_mm_yyyy_to_cic_day` → `training_day` / `testing_day`.
- Labels **`ddos`** if the PCAP’s **[first, last] time span** overlaps any configured **SYN** local-time window for that day; else **`normal`**.
- On any failure (bad pcap, unknown date, missing config), **falls back** to `syn_count` with the same threshold.

**Timezone:** windows are interpreted in the JSON `timezone` (default `America/Halifax`). Adjust if your PCAP timestamps do not align with CIC’s published local schedule.

**File-level limitation:** one label per whole PCAP. Long captures spanning benign + attack periods may be ambiguous; split PCAPs or accept mixed content if needed.

**Regenerating data:** after changing labeling, re-run conversion and rebuild `train`/`val`/`test` splits (`scripts/split_train_val_test.py`).

### 2) PCAP -> images (this changes model input distribution)
File: `scripts/pcap_to_images.py`
Conversion function:
- `packet_to_image_224x224(pkt_bytes, syn_only)`
  - If `syn_only` is enabled and packet is not TCP SYN: return `None` (skip)
  - Otherwise:
    - Convert raw bytes to `uint8` array `arr`
    - Create smallest square image that fits all bytes (`n = ceil(sqrt(len(arr)))`)
    - Pad with zeros up to `n*n` (or truncate if too large)
    - Reshape to `(n, n)` grayscale
    - Resize to `(224, 224)` using PIL `Image.Resampling.LANCZOS`
    - Return `np.array(pil_img)` (2D grayscale image)

Conversion IO:
- The script iterates packets from a PCAP (`iter_pcap_packets()`).
- It saves PNG images to `--out`:
  - filename: `{prefix}_{saved}.png`
  - stops after `--max-images` (default `200`)

### 3) How ddos vs normal PCAPs are converted in this repo
File: `scripts/convert_pcaps_to_images.sh`

Env:
- `CLASSIFY_METHOD` — `syn_count` (default) or `cic_schedule`
- `CIC_SCHEDULE_CONFIG` — path to JSON (default: `scripts/cic_ddos2019_syn_windows.json`)

For each PCAP:
- classify PCAP first using `scripts/classify_pcap_syn.py` (per env above)
- runs converter with **no** `--syn-only` (convert all packets) for both classes

Images are stored under `IMG_ROOT`, typically:
- `…/images/train/ddos/`
- `…/images/train/normal/`

Server batch: `scripts/process_zips_in_batches.sh` calls the same converter. It defaults to **`CLASSIFY_METHOD=cic_schedule`** and processes **both** capture days (all `PCAP-01-12_*.zip` + `PCAP-03-11.zip`). Set `CLEAN_TRAIN_IMAGES_FIRST=1` once to wipe old `train/ddos` and `train/normal` PNGs before a full regeneration.

**Server (91.98.68.228) — typical flow**

1. Copy updated repo (or `git pull`) under `~/huawei` so `scripts/cic_ddos2019_syn_windows.json`, `classify_pcap_syn.py`, and `pcap_time.py` exist.
2. Use the project venv for NumPy and Python deps, e.g. `export PYTHON="$HOME/huawei/.venv/bin/python"` (adjust path if root: `/root/huawei/.venv/bin/python`).
3. Run from repo root:

```bash
export PYTHON=/path/to/huawei/.venv/bin/python
export CLASSIFY_METHOD=cic_schedule
export CLEAN_TRAIN_IMAGES_FIRST=1   # first run after label change; omit later if appending
cd ~/huawei && bash scripts/process_zips_in_batches.sh
```

4. **Rsync** `~/datasets/CICDDoS2019/images/train/` to your laptop `dataset/images/train/` when done.

**Visual QC (qualitative):** True high-volume SYN-flood windows often produce **dense, high-frequency / “noisy”** packet images; **false-positive “ddos”** labels (e.g. from old SYN-threshold rules) can look **smoother / blob-like** next to them. Use this only as a sanity check—labels from **`cic_schedule`** should reduce mismatch between attack time and image content.

### 4) Image preprocessing expected by the model
File: `notebook/train_resnext.py` (`make_transforms()`)

Training/eval transforms (key parts):
- Decode image
- `to_rgb_np()`:
  - if image is grayscale (2D): replicate to 3 channels
  - if 1 channel: repeat to 3 channels
- Resize/Crop:
  - Train: RandomResizedCrop(224) and RandomHorizontalFlip (if augment enabled)
  - Eval/Test: resize to `resize_size` (default 256), then CenterCrop(224), then normalize
- Rescale: `vision.Rescale(1/255)`
- Normalize: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Convert HWC -> CHW (`vision.HWC2CHW()`)

