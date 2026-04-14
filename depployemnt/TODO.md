# Deployment TODO

## Next Steps

### 1. Server environment setup
- [x] Update system packages (`apt update && apt upgrade -y`)
- [x] Install Python 3 and pip (or use venv)
- [x] Install project dependencies (MindSpore CPU, mindcv, numpy, PIL, etc.)
- [x] Clone or copy this repo to the server

### 2. Model and pipeline deployment
- [x] Transfer model checkpoint (`model/resnext50_32x4d_best.ckpt`) to server
- [x] Deploy PCAP→image conversion code (`scripts/pcap_to_images.py`)
- [x] Deploy inference script (load model, preprocess images, run Option 3 fusion)
- [x] Verify inference runs correctly with a test PCAP or sample images

### 3. Packet capture (live traffic)
- [x] Install packet capture tools (e.g., `tcpdump`, or `libpcap`-based tools)
- [x] Configure capture to run on the server's interface(s) (`eth0` via `run_pipeline.py --live --interface eth0`)
- [x] Implement time-based windowing (e.g., 2-second chunks) for streaming (`--window-sec 2`)

### 4. End-to-end pipeline
- [x] Wire capture → conversion (both views) → inference → alert/log (`synflood-detector.service`)
- [x] Add queue/drop policy if conversion or inference lags
- [x] Tune image cap per window (start with lower than 200 if CPU struggles)

### 5. Operations
- [x] Set up logging and basic monitoring (`/var/log/synflood-detector/detections.log`, `journalctl -u synflood-detector`)
- [x] Document how to start/stop the pipeline — see `depployemnt/OPERATIONS.md`
- [ ] (Optional) Restrict security group inbound rules to trusted IPs (per Huawei warning)

---

## Reference docs in this folder
- `README.md` — Overview and Option 3 summary
- `option3_streaming_plan.md` — Two-branch inference and fusion logic
- `dataset_and_conversion_reference.md` — Exact conversion logic to match
- `preprocess_matches_repo_eval.md` — Image preprocessing for inference
- `window_to_images_mapping.md` — How windows map to image batches
