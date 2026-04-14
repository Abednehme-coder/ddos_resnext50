# Operations (ECS live pipeline)

## Which host is which

| Role | Address | Notes |
|------|---------|--------|
| **Huawei ECS (this doc)** | **`213.250.144.74`** | `synflood-detector`, `/opt/synflood-detector/`, live inference — **deploy checkpoints here** |
| Dataset / PCAP processing | `91.98.68.228` | Training data, zips, **not** the ECS detector layout |

SSH examples: `ssh root@213.250.144.74` (adjust user if you use `ubuntu@`).

## What runs on the server

- **Service:** `synflood-detector.service` — **canonical unit file in this repo:** `depployemnt/systemd/synflood-detector.service` (must stay aligned with `run_pipeline.py` flags).
- **Pipeline:** **single-view**, **all-packets** → one image per packet (up to cap), **mean softmax** over the window, then `ddos` vs `normal`. (Older docs referred to “Option 3” two-branch fusion; **current code** is this single path only.)
- **Process:** `/opt/synflood-detector/venv/bin/python .../run_pipeline.py --live --ckpt .../model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt --interface eth0 --window-sec 0.3 --max-images-per-window 50 ...` (canonical flags in `depployemnt/systemd/synflood-detector.service`)
- **Capture:** `tcpdump` on `eth0`, rotating PCAPs under `/var/capture/`. On many Linux images `tcpdump` **drops privileges** to user `tcpdump` when writing `-w` files, so `/var/capture` must be **`chown tcpdump:tcpdump`** (not `root:root` only). If you see `tcpdump stopped (rc=1)` and `Permission denied` in `/var/log/synflood-detector/tcpdump.err.log`, fix with: `sudo mkdir -p /var/capture && sudo chown tcpdump:tcpdump /var/capture`. `install_ecs_systemd.sh` does this automatically.
- **Log:** one JSON line per processed window in `/var/log/synflood-detector/detections.log`

### Refresh systemd unit on ECS (after repo changes)

If `run_pipeline.py` arguments change, update `depployemnt/systemd/synflood-detector.service`, then:

```bash
bash depployemnt/install_ecs_systemd.sh
```

This copies the unit to `/etc/systemd/system/`, runs `daemon-reload`, and restarts the service.

### Sync live pipeline code to ECS (minimal)

Only the files the detector **imports** are copied: `run_pipeline.py` and `scripts/pcap_to_images.py` (capture → per-packet images → batch inference → log). Training scripts, notebooks, and datasets are **not** synced. **Checkpoints** stay on the server unless you run `deploy_ckpt_to_ecs.sh`.

```bash
bash depployemnt/sync_pipeline_to_ecs.sh
```

Defaults: **`root@213.250.144.74`**, destination **`/opt/synflood-detector/repo/`**. Optional: `NO_RESTART=1`, `ECS_HOST=ubuntu@213.250.144.74`.

**Time window:** The repo unit file uses **`--window-sec 0.3`** to match **`per_second_0p3_focal_v2`** training. After changing the unit, run `bash depployemnt/install_ecs_systemd.sh` on ECS. For a 2 s window instead, edit the unit and redeploy.

## Commands

```bash
# Status
sudo systemctl status synflood-detector

# Start / stop / restart
sudo systemctl start synflood-detector
sudo systemctl stop synflood-detector
sudo systemctl restart synflood-detector

# Logs (service stdout/stderr)
sudo journalctl -u synflood-detector -f

# Detection log (JSON lines)
sudo tail -f /var/log/synflood-detector/detections.log
```

## Boot on startup

On the deployed ECS instance this is already enabled. On a fresh host:

```bash
sudo systemctl enable synflood-detector
```

## Deploy a new checkpoint (from your laptop)

Default artifact for the 0.3 s same-domain run: `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` in the repo.

From the repo root (defaults to **`root@213.250.144.74`**):

```bash
bash depployemnt/deploy_ckpt_to_ecs.sh
```

If your login user differs: `ECS_HOST=ubuntu@213.250.144.74 bash depployemnt/deploy_ckpt_to_ecs.sh`

This copies the checkpoint to `/opt/synflood-detector/repo/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (with a timestamped `.bak` of the previous file), then restarts `synflood-detector`.

Overrides:

```bash
LOCAL_CKPT=/path/to/other.ckpt REMOTE_DIR=/opt/synflood-detector/repo/model NO_RESTART=1 bash depployemnt/deploy_ckpt_to_ecs.sh
```

**Note:** Live `run_pipeline.py` builds **per-packet** images from PCAPs, while the 0.3 s model was trained on **time-window** images. Behavior may differ from offline val/test; monitor `detections.log` and tune `--max-images-per-window` if needed.

## Offline test (no live capture)

```bash
source /opt/synflood-detector/venv/bin/activate
python /opt/synflood-detector/repo/run_pipeline.py \
  --pcap /path/to/file.pcap \
  --ckpt /opt/synflood-detector/repo/model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --max-images-per-window 50 \
  --log-file /tmp/synflood_test.log
```

## Webhook alerts (optional)

Pass `--notify-url https://...` to `run_pipeline.py` (or add to the systemd `ExecStart` line). **As of the current repo**, webhook POST runs only when the log line has **`"alert": true`** (not on every `pred: ddos`). Alerts require:

- `p_ddos >= --alert-min-p-ddos` (default `0.0`)
- that condition holding for **`--alert-consecutive`** consecutive windows (default `1`)

Example stricter policy: `--alert-min-p-ddos 0.8 --alert-consecutive 2`.

Every window still appends a JSON line to `detections.log` with `pred`, `p_ddos`, `streak`, `alert`, `alert_min_p_ddos`, `alert_consecutive`.

## Offline benchmark (0.3 s model, local)

Run from repo root (writes metrics to console; full record in `reports/eval_results_per_second_0p3.md`):

```bash
bash scripts/run_all_eval_0p3.sh
# Optional: DEVICE=GPU bash scripts/run_all_eval_0p3.sh
```

This evaluates `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` on `dataset/images_per_second_window_0p3` (val, test, and blind test).

**Snapshot (2026-04-03, CPU):** test accuracy **98.10%**, macro F1 **0.937**; blind test (270 images, seed 42) **98.9%** accuracy, macro F1 **0.989**. See the report file for confusion matrices.

## Online verification (ECS)

After deploy, confirm the service and recent JSON lines:

```bash
ssh root@213.250.144.74 "systemctl is-active synflood-detector && tail -n 5 /var/log/synflood-detector/detections.log"
```

**Last checked:** 2026-04-03 — service `active`, checkpoint present, `detections.log` updating. Live inference uses per-packet images inside each window (see note in `reports/eval_results_per_second_0p3.md`).
