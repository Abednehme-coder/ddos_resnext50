# Offline evaluation: `per_second_0p3_focal_v2` (0.3 s time-window images)

**Date:** 2026-04-03  
**Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`  
**Data:** `dataset/images_per_second_window_0p3/` (train/val/test from same 0.3 s distribution)  
**Device:** **GPU** (NVIDIA RTX 3070, driver 535.x, CUDA 11.6 paths as in `run_all_eval_0p3.sh`). `bash scripts/run_all_eval_0p3.sh` completed 2026-04-03; metrics match prior CPU numbers.

**Calibration / figures:** `bash scripts/run_all_paper_figures.sh` → `research/figures/*.pdf`; see `reports/paper_figures_2026-04-03.log` (test ECE ≈ 0.074, heuristic best threshold ≈ 0.87 for max F1).

**Cross-domain:** `reports/cross_domain_eval_2026-04-03.md` (0.3 s ↔ per-packet test; **not** comparable to same-domain metrics).

**Environment pin:** `reports/REPRODUCIBILITY.md`, `reports/pip_freeze_2026-04-03.txt`.

**Checkpoint policy:** `scripts/eval_test_detailed.py`, `scripts/eval_roc_pr_calibration.py`, and `scripts/blind_test_random.py` only accept the primary 0.3 s focal-v2 file under `model/per_second_0p3_focal_v2/` (see `scripts/eval_primary_ckpt.py`). Override only if you must: `HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1`. `scripts/eval_per_second_1s_model.sh` sets this for the non-primary 1 s experiment.

## `scripts/eval_test_detailed.py`

### Validation split

- Samples: 1842  
- Overall accuracy: **0.982085**  
- Confusion matrix [rows=true, cols=pred]:

|        | ddos | normal |
|--------|-----:|-------:|
| ddos   |  133 |      0 |
| normal |   33 |   1676 |

- Per-class F1: ddos **0.889632**, normal **0.990251**  
- Macro F1: **0.939942**  
- Weighted F1: **0.982986**  
- Balanced accuracy: **0.990345**

### Test split

- Samples: 1845  
- Overall accuracy: **0.981030**  
- Confusion matrix:

|        | ddos | normal |
|--------|-----:|-------:|
| ddos   |  135 |      0 |
| normal |   35 |   1675 |

- Per-class F1: ddos **0.885246**, normal **0.989660**  
- Macro F1: **0.937453**  
- Weighted F1: **0.982020**  
- Balanced accuracy: **0.989766**

## `scripts/blind_test_random.py` (test split, seed 42)

- **Total:** 270 images (135 per class — maximum equal split given class counts on test).  
- Accuracy: **267/270 = 98.9%**  
- Macro F1: **0.9889**  
- Confusion matrix:

|        | ddos | normal |
|--------|-----:|-------:|
| ddos   |  135 |      0 |
| normal |    3 |    132 |

## Reproduce

```bash
cd /path/to/huawei
./scripts/run_all_eval_0p3.sh
# Or individually:
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

## Online (ECS) verification — 2026-04-03

- **Host:** `213.250.144.74`  
- **Service:** `synflood-detector` — **active**  
- **Checkpoint on server:** `/opt/synflood-detector/repo/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (deploy with `depployemnt/deploy_ckpt_to_ecs.sh`)  
- **Sample `detections.log`:** JSON lines with `pred`, `p_ddos`, `images_all` (live traffic; labels not available without ground truth).

**Note:** Live `run_pipeline.py` uses **per-packet** images from each PCAP window, not the offline **aggregated** 0.3 s training images. Offline metrics above are the authoritative benchmark for the 0.3 s trained model; monitor production logs after aligning `--window-sec` with training (see `depployemnt/systemd/synflood-detector.service`).

### Research report

The LaTeX write-up `research/main.tex` summarizes these offline numbers, adds a **test matrix** (Table: conducted tests), and explains that **many `"pred": "ddos"` lines on a public ECS do not imply a confirmed attack**—they are model outputs on unlabeled traffic (possible false positives / domain shift).
