# Appendix — Huawei ICT Innovation (Regional)

**Companion to:** `reports/ICT_Innovation_Regional_Report.md`

This appendix holds extended metrics, reproduction commands, script inventory, cross-domain evaluation, and environment pins. The main report should remain readable on its own; judges who want depth can read this document in full.

---

## A. Document map

| File | Purpose |
|------|---------|
| `SYSTEM_DESCRIPTION.txt` | Canonical internal system description (repo root) |
| `reports/eval_results_per_second_0p3.md` | Primary 0.3 s evaluation log and commands |
| `reports/cross_domain_eval_2026-04-03.md` | Cross-domain (representation mismatch) study |
| `reports/REPRODUCIBILITY.md` | Git commit, Python, MindSpore, GPU snapshot |
| `reports/pip_freeze_2026-04-03.txt` | Frozen dependencies (if present) |
| `reports/paper_figures_2026-04-03.log` | Figure/calibration run log |

---

## B. Reproducibility snapshot (2026-04-03)

| Item | Value |
|------|--------|
| Git commit | `96b82d4cf152053175950a346bdd204b6e109ebf` |
| Python | 3.10.12 |
| MindSpore | 2.6.0 |
| mindcv | 0.3.0 |
| GPU (eval machine) | NVIDIA GeForce RTX 3070 |
| NVIDIA driver | 535.288.01 |
| CUDA toolkit (typical for scripts) | `/usr/local/cuda-11.6` |

Full pip freeze: regenerate with `.venv/bin/pip freeze > reports/pip_freeze_YYYY-MM-DD.txt` or use `reports/pip_freeze_2026-04-03.txt` if checked in.

**MindSpore install:** Always use the official command matrix from [MindSpore install](https://www.mindspore.cn/install/en) matching OS, Python, and CUDA; then `pip install -r requirements.txt` for `mindcv`, `numpy`, `Pillow`.

---

## C. Data processing pipeline (detailed)

1. **Source archives:** CICDDoS2019 zips processed in batches (e.g. `scripts/process_zips_in_batches.sh`).  
2. **PCAP labeling:** `scripts/classify_pcap_syn.py` — SYN-oriented heuristic; PCAPs with **>100 TCP SYN packets** treated as DDoS, else normal (as documented in `SYSTEM_DESCRIPTION.txt`).  
3. **Packet selection:** For DDoS PCAPs, **SYN-only** packet conversion (`--syn-only`); for normal, all packets (as in system description).  
4. **Image conversion:** `scripts/pcap_to_images.py` — raw bytes → 2D grayscale → **224×224**; batch driver `scripts/convert_pcaps_to_images.sh`.  
5. **Layout:** Class folders under `train/`, `val/`, `test/` with `ddos/` and `normal/`.  
6. **Time-window variant:** Images under `dataset/images_per_second_window_0p3/` built with the project’s per-window rules (0.3 s); this is the **primary** training distribution for the regional report.

---

## D. Training configuration (as documented)

**Entry point:** `notebook/train_resnext.py`

**Model:** ResNeXt-50 32×4d via mindcv.

**Preprocessing (summary):**

- Train: `RandomResizedCrop(224)`, `RandomHorizontalFlip`.  
- Eval/test: Resize 256 → `CenterCrop` 224.  
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

**Documented run (primary narrative):**

- Framework: MindSpore (Graph mode in documented configuration).  
- Optimizer: Adam / AdamWeightDecay; lr **1e-3**, weight decay **1e-4**.  
- Batch size **16**, epochs **8**.  
- Loss: weighted cross-entropy (class weights from training counts).  
- Checkpoint selection: **macro F1** on validation.

**Checkpoints:**

| Role | Path |
|------|------|
| Primary (0.3 s) | `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` |
| Legacy per-packet | `model/resnext50_32x4d_best.ckpt` (and numbered variants noted in `SYSTEM_DESCRIPTION.txt`) |

**Checkpoint safety:** Some eval scripts default to the primary checkpoint path; overrides may require `HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1` (see `scripts/eval_primary_ckpt.py`).

---

## E. Primary evaluation — extended tables (0.3 s)

Source: `scripts/eval_test_detailed.py`, `reports/eval_results_per_second_0p3.md`.

### Validation (1842 samples)

- Overall accuracy: **0.982085**  
- Macro F1: **0.939942**  
- Weighted F1: **0.982986**  
- Balanced accuracy: **0.990345**

Confusion [rows=true, cols=pred]:

|        | ddos | normal |
|--------|-----:|-------:|
| ddos   |  133 |      0 |
| normal |   33 |   1676 |

Per-class F1: ddos **0.889632**, normal **0.990251**.

### Test (1845 samples)

- Overall accuracy: **0.981030**  
- Macro F1: **0.937453**  
- Weighted F1: **0.982020**  
- Balanced accuracy: **0.989766**

Confusion:

|        | ddos | normal |
|--------|-----:|-------:|
| ddos   |  135 |      0 |
| normal |   35 |   1675 |

Per-class F1: ddos **0.885246**, normal **0.989660**.

### Blind test (270 images, seed 42)

- Accuracy **267/270 (98.9%)**, macro F1 **0.9889**  
- Confusion: ddos 135/0; normal 3/132  

### Reproduction commands (GPU example)

```bash
cd /path/to/huawei
bash scripts/run_all_eval_0p3.sh
```

Individual calls (adjust venv path):

```bash
.venv/bin/python scripts/eval_test_detailed.py \
  --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --data-root dataset/images_per_second_window_0p3 --split val --device-target GPU

.venv/bin/python scripts/eval_test_detailed.py \
  --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --data-root dataset/images_per_second_window_0p3 --split test --device-target GPU

.venv/bin/python scripts/blind_test_random.py \
  --data-root dataset/images_per_second_window_0p3 \
  --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --split test --total 270 --seed 42 --device-target GPU
```

---

## F. Legacy per-packet evaluation (reference)

- Test macro F1 ≈ **95.56%**, test accuracy ≈ **95.56%** (older ResNeXt run on `dataset/images`).  
- Small blind test: 5+5 images, seed 42 → **10/10** correct.  
- Large blind test: **3000** images (1500 per class), shuffled — confusion:

|        | pred ddos | pred normal |
|--------|----------:|------------:|
| true ddos | 1498 | 2 |
| true normal | 0 | 1500 |

Accuracy **2998/3000 (99.9%)**, macro F1 **0.9993**.

Example command:

```bash
python scripts/blind_test_random.py --data-root ./dataset/images --split test --total 3000
```

---

## G. Cross-domain evaluation (representation mismatch)

**Purpose:** Show that a model trained on one image construction **does not** reliably work on another without adaptation.

**Recorded:** 2026-04-03, `scripts/eval_cross_domain.py --device-target GPU`. Full narrative: `reports/cross_domain_eval_2026-04-03.md`.

### G.1 — 0.3 s checkpoint on per-packet test split

- Train order root: `dataset/images_per_second_window_0p3/train`  
- Eval images: `dataset/images/test`  
- Samples: **19300**  
- Overall accuracy: **0.987565** (majority normal)  
- Macro F1: **0.496872**  
- Balanced accuracy: **0.500000**  

Argmax **never** predicted DDoS on this eval set (DDoS recall **0** in confusion table). **Headline accuracy is misleading** here.

### G.2 — Per-packet checkpoint on 0.3 s test split

- Checkpoint: `model/resnext50_32x4d_best.ckpt`  
- Train order: `dataset/images/train`  
- Eval: `dataset/images_per_second_window_0p3/test`, **1845** samples  
- Overall accuracy: **0.898645**  
- Macro F1: **0.473309**  
- DDoS recall **0** under argmax; some normal windows predicted as DDoS.

### Reproduce

```bash
cd /path/to/huawei
export CUDA_HOME=/usr/local/cuda-11.6   # adjust to your install
export PATH="$CUDA_HOME/bin:$PATH"
# Set LD_LIBRARY_PATH per MindSpore GPU docs

.venv/bin/python scripts/eval_cross_domain.py \
  --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --train-order-root dataset/images_per_second_window_0p3 \
  --eval-root dataset/images --split test --device-target GPU

.venv/bin/python scripts/eval_cross_domain.py \
  --ckpt model/resnext50_32x4d_best.ckpt \
  --train-order-root dataset/images \
  --eval-root dataset/images_per_second_window_0p3 --split test --device-target GPU
```

---

## H. Figures and calibration

- **ROC/PR/calibration:** `bash scripts/run_all_paper_figures.sh` → output under `research/figures/` (PDFs).  
- **Log:** `reports/paper_figures_2026-04-03.log` notes e.g. test ECE ≈ **0.074**, heuristic best threshold ≈ **0.87** for max F1 (confirm on your machine if re-run).

---

## I. Script and module index

| Path | Role |
|------|------|
| `notebook/train_resnext.py` | Training and standard evaluation entry |
| `scripts/classify_pcap_syn.py` | PCAP labeling (SYN heuristic) |
| `scripts/pcap_to_images.py` | Packet bytes → 224×224 image |
| `scripts/convert_pcaps_to_images.sh` | Batch conversion driver |
| `scripts/process_zips_in_batches.sh` | Batch unzip/process CICDDoS2019 |
| `scripts/blind_test_random.py` | Blind random-sample evaluation |
| `scripts/eval_test_detailed.py` | Detailed val/test metrics |
| `scripts/eval_cross_domain.py` | Cross-domain (mismatched representation) |
| `scripts/eval_roc_pr_calibration.py` | ROC/PR/calibration |
| `scripts/eval_primary_ckpt.py` | Primary checkpoint policy helper |
| `scripts/run_all_eval_0p3.sh` | One-shot primary eval |
| `scripts/run_all_paper_figures.sh` | Figure generation |
| `run_pipeline.py` | Live/offline pipeline (note: per-packet windows vs aggregated 0.3 s training — align for production) |
| `depployemnt/` | Deployment helpers (spelling as in repo) |

---

## J. Repository layout (abbreviated)

```
huawei/
├── notebook/train_resnext.py
├── scripts/
├── model/
│   ├── per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt
│   └── resnext50_32x4d_best.ckpt   # legacy per-packet
├── dataset/
│   ├── images/                      # per-packet layout
│   └── images_per_second_window_0p3/
├── reports/                       # this appendix lives here
├── research/                      # paper assets, figures
├── requirements.txt
├── README.md
└── SYSTEM_DESCRIPTION.txt
```

---

## K. Glossary

| Term | Meaning |
|------|---------|
| PCAP | Packet capture file |
| SYN flood | TCP half-open / handshake exhaustion attack pattern |
| Macro F1 | Unweighted mean of per-class F1 scores |
| Time-window image | Aggregate of packets in a short time window (here 0.3 s) into one image |
| Per-packet image | One image per packet (legacy setup in this repo) |
| Blind test | Inference on randomly sampled files with shuffled order; model inputs are images only |

---

## L. Online / deployment note (optional)

If the team maintains a remote demo, document **how judges access it** (URL, test PCAP, expected JSON fields) in a **separate non-public** handout if credentials are involved. The internal eval note in `reports/eval_results_per_second_0p3.md` references an ECS deployment and log format (`pred`, `p_ddos`, etc.); mirror that description in your demo guide without committing secrets to a public repository.

**Important:** Many consecutive `"pred": "ddos"` lines on **unlabeled** live traffic do **not** prove an attack—they are model scores subject to domain shift and thresholding.

---

*End of appendix.*
