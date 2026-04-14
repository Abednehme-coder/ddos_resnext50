# Deployment Progress

## Completed

### 1. Planning
- [x] Documented training dataset formation and PCAP→image conversion logic
- [x] Chose Option 3: two-branch inference (all_packets + syn_only) with fused predictions
- [x] Defined window parameters: time-based (2s default), max 200 images per view per window
- [x] Aligned preprocessing with repo eval pipeline (resize 256, center-crop 224, ImageNet normalize)

### 2. Huawei Cloud Setup
- [x] Redeemed experience coupon `CP260303034531EVAV` ($40 USD)
- [x] Created ECS instance:
  - **Region:** TR-Istanbul
  - **Specs:** General computing-basic | t6.large.2 | 2 vCPUs | 4 GiB RAM
  - **Disk:** General Purpose SSD 40 GiB
  - **OS:** Ubuntu 24.04 server 64bit
  - **EIP:** Bandwidth 1 Mbit/s (yearly/monthly billing)
  - **Duration:** 1 month, auto-renew off
- [x] Established SSH connection to the server
- **ECS public IP (live detector):** `213.250.144.74` — **not** the separate dataset host (`91.98.68.228`)

### 3. Constraints
- No GPU available in selected instance catalog; inference will run on CPU
- Coupon restricts some services; ECS was confirmed usable at checkout

### 4. Live deployment (completed)
- **MindSpore CPU** + `mindcv` in `/opt/synflood-detector/venv`
- **Repo:** `/opt/synflood-detector/repo` (includes `run_pipeline.py`, `scripts/`)
- **Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (same path on ECS; deploy via `depployemnt/deploy_ckpt_to_ecs.sh`)
- **systemd:** `synflood-detector.service` — `tcpdump` on `eth0`, **0.3 s** PCAP windows (aligned with `per_second_0p3_focal_v2` training), **single-view all-packets** inference; canonical unit: `depployemnt/systemd/synflood-detector.service`
- **Logs:** `/var/log/synflood-detector/detections.log` (see `OPERATIONS.md`)

### 5. Evaluation (2026-04-03)
- [x] Offline: `scripts/run_all_eval_0p3.sh` — val/test + blind (270) on `images_per_second_window_0p3`; see `reports/eval_results_per_second_0p3.md`
- [x] Online: ECS service `active`, `detections.log` updating (`213.250.144.74`)
- [x] **systemd on ECS** refreshed with repo unit (`bash depployemnt/install_ecs_systemd.sh`) — live **`--window-sec 0.3`** active as of 2026-04-03
- [x] **Cross-domain** GPU runs logged in `reports/cross_domain_eval_2026-04-03.md` (train/serve representation mismatch → collapsed DDoS F1; not a production substitute for same-domain metrics)
- [x] **Paper figures:** `bash scripts/run_all_paper_figures.sh` → `research/figures/*.pdf`; log `reports/paper_figures_2026-04-03.log`
- [x] **Reproducibility:** `reports/REPRODUCIBILITY.md`, `reports/pip_freeze_2026-04-03.txt`
- [x] **LaTeX:** `research/main.tex` updated (cross-domain table, env table, calibration note, limitations/conclusion)

### 6. ECS alignment (2026-04-03, automated)
- [x] **`deploy_ckpt_to_ecs.sh`** — uploaded `per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` to `/opt/synflood-detector/repo/model/per_second_0p3_focal_v2/`
- [x] **`install_ecs_systemd.sh`** — live unit now uses `--ckpt .../model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (verified `systemctl status`)
- [x] **Dev venv:** `pytest` installed; `pytest tests/` — 7 passed

**PDF build:** `pdflatex` is not installed on this dev host (no passwordless `sudo` for `apt install texlive`). Build `research/main.tex` on a machine with TeX or Overleaf.

---

*Last updated: ECS checkpoint + systemd canonical path; venv pytest (2026-04-03).*
