## depployemnt

Notes and requirements for deploying the trained ResNeXt (MindSpore) SYN-flood detector on Huawei Cloud.

### What the model actually learns in this repo
This project detects `ddos` vs `normal` using **packet-derived images** and a ResNeXt-50 backbone.

Training pipeline summary (from this repo):
1. **PCAP labeling (offline only):** a PCAP is labeled `ddos` if TCP SYN packets count `> SYN_THRESHOLD` (default `100`), else `normal`.
2. **PCAP -> images (this affects model inputs):**
   - If PCAP label is `ddos`: convert **all packets**.
   - If PCAP label is `normal`: convert **all packets**.
3. **Train:** MindSpore dataset loads these images from `dataset/train/{ddos,normal}` and applies ImageNet normalization + RGB conversion.

### Real-time pipeline (current code: `run_pipeline.py`)
Streaming does **not** use offline PCAP labels. For each **tcpdump** window PCAP:

1. Convert **all packets** (up to `--max-images-per-window`) into **224×224** grayscale images, then **one** forward pass per batch; **mean** of softmax probabilities over images → `ddos` vs `normal`.

Older notes describe a **two-branch “Option 3”** design; **the deployed implementation** is the **single-view** path above. See `depployemnt/option3_streaming_plan.md` for history only.

### Progress & next steps
- **PROGRESS.md** — What’s done (ECS provisioned, SSH connected, etc.)
- **TODO.md** — Step-by-step checklist for deployment
- **OPERATIONS.md** — Hosts, systemd, logs, **`sync_pipeline_to_ecs.sh`** (runtime Python only), deploy checkpoint, optional webhook
- **`depployemnt/systemd/synflood-detector.service`** — Canonical `ExecStart` (keep in sync with `run_pipeline.py --help`)

