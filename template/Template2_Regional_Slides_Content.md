# Template 2 (Regional) — Slide Content Draft

Use this as your master content, then paste into:
- `[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage) .pptx`

---

## Slide 1 — Title
**SYNSheild: Huawei MindSpore + ResNeXt-50 for Early SYN-Flood DDoS Detection**

Huawei ICT Competition — Innovation Track (Regional Stage)

- Team: Wafik Ibrahim, Mohammed Farhat, Abed Al Rida Nehme
- Instructor: Ali Nassar
- Institution: Beirut Arab University (BAU), Lebanon
- Date: 11 April 2026

Speaker note:
- "We built an end-to-end, reproducible AI pipeline that detects SYN-flood DDoS from packet-window images with high macro-F1 and practical deployment pathways."

---

## Slide 2 — Problem & Application Value (35%)
- SYN-flood DDoS attacks reduce availability of critical services
- Traditional threshold/signature defenses are fast but can be brittle in dynamic traffic
- Security teams need practical AI support with:
  - high detection quality
  - reproducible evaluation
  - clear deployment path

Why this matters regionally:
- Universities, enterprises, and public services in our region need resilient network availability

---

## Slide 3 — Existing Methods vs Our Approach
### Common methods today
1. Rule-based / signatures
   - + simple, lightweight
   - - weak generalization to traffic variation
2. Classical ML on handcrafted features
   - + explainable features
   - - heavy feature engineering effort
3. Prior CNN packet-image methods
   - + better automatic pattern learning
   - - often not integrated with Huawei AI ecosystem or full reproducibility artifacts

### Our approach
- Packet bytes -> grayscale image windows -> ResNeXt-50 in MindSpore
- End-to-end scripts from dataset processing to blind testing

---

## Slide 4 — Core Innovation (40%)
**What is innovative in our submission:**
- Architecture upgrade: ResNet-style baseline -> **ResNeXt-50 (32x4d)**
- Representation: primary model on **0.3 s time-window** images
- Training strategy: class imbalance handling, macro-F1-driven checkpoint selection
- Engineering innovation: full automation pipeline (zip -> pcap -> image -> train -> eval)
- Evaluation rigor: detailed split metrics, blind test, calibration and cross-domain checks

One-line value:
- "Innovation is not only model choice; it is the complete, verifiable system around it."

---

## Slide 5 — Huawei Ecosystem Compatibility & Role
- Framework: **Huawei MindSpore 2.6.0**
- Model tooling: **mindcv** for ResNeXt backbone integration
- Graph-execution oriented pipeline with reproducible scripts
- Clear compatibility narrative for Huawei ecosystem adoption:
  - Ascend-ready software ecosystem direction
  - Huawei cloud / edge deployment pathway
  - Competition-aligned technology stack

Huawei role in our project:
- MindSpore enabled the full training + evaluation implementation we present today

---

## Slide 6 — End-to-End Technical Flow
PCAP traffic -> SYN-oriented labeling for dataset construction -> 224x224 grayscale images -> ResNeXt-50 classifier -> DDoS/Normal decision

Implementation modules:
- `scripts/classify_pcap_syn.py`
- `scripts/pcap_to_images.py`
- `notebook/train_resnext.py`
- `scripts/eval_test_detailed.py`
- `scripts/blind_test_random.py`

---

## Slide 7 — Performance Highlights (Primary 0.3 s Model)
Checkpoint:
- `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`

Results:
- Validation: 98.21% accuracy, 0.9399 macro-F1
- Test: 98.10% accuracy, 0.9375 macro-F1
- Blind test (270 balanced): 98.9% accuracy, 0.9889 macro-F1

Message to judges:
- We report both accuracy and macro-F1 to avoid imbalance bias

---

## Slide 8 — ROC/PR + Confusion Insights
Use your generated figures:
- ROC: `research/figures/roc_test.pdf`
- PR: `research/figures/pr_test.pdf`

Key points:
- Strong separability in test ROC/PR
- Confusion pattern shows robust DDoS detection in same-domain setup
- Most residual errors are operationally manageable through threshold tuning and alert workflows

---

## Slide 9 — Honest Comparison & Scientific Rigor
What we compare against:
- Trivial majority baseline (shows why raw accuracy can mislead)
- Prior ResNet packet-image literature baseline context

What we additionally did (often missing in similar projects):
- Cross-domain evaluation (0.3 s <-> per-packet mismatch)
- Explicit reporting that representation mismatch hurts transfer

Judge-facing message:
- We prefer transparent, deployment-relevant science over inflated claims

---

## Slide 10 — Deployment Strategy (Server + Network Devices)
### A) Server deployment (available now)
- Inference service on server for PCAP or streamed captures
- Batch/near-real-time detection pipeline with logs and probabilities

### B) Early detection near network edge (switches/routers)
Practical architecture:
- Switch/router exports telemetry (NetFlow/sFlow/mirrored packets)
- Lightweight feature/image builder at edge collector
- Model inference on nearby compute node (server/edge AI box)
- Alerts returned to SOC / controller for mitigation actions

Important clarification:
- Most routers/switches do not run full CNN inference natively.
- Recommended design: telemetry from devices + inference on attached compute for early warning.

---

## Slide 11 — Requirements for Optimal Operation
### Training / evaluation (recommended)
- GPU: NVIDIA class (tested on RTX 3070)
- CUDA: ~11.6-compatible stack
- Python: 3.10.x
- MindSpore: 2.6.0
- mindcv + numpy + pillow

### Inference deployment
- CPU possible for low-throughput scenarios
- GPU/accelerator preferred for higher throughput and lower latency
- Stable storage and logging for audit/reproducibility

Operational requirements:
- Keep train-time and deploy-time representation aligned (same windowing/image construction)

---

## Slide 12 — Challenge Faced: No GPU on Server
Problem we encountered:
- Some server environments lacked GPU acceleration

Impact on work:
- Slower experimentation and iteration cycles
- Higher inference latency under heavier loads
- More difficult rapid threshold/calibration sweeps in production-like conditions

How we mitigated it:
- Kept authoritative offline benchmarks from GPU environment
- Used CPU-compatible evaluation where needed
- Separated validated offline metrics from live unlabeled traffic interpretation

---

## Slide 13 — Enhancement Roadmap
Near-term enhancements:
- Align live pipeline windowing exactly with training representation
- Add latency/throughput benchmark tables (CPU vs GPU vs edge)
- Improve thresholding policy by environment

Mid-term enhancements:
- Multi-attack-class detection (beyond SYN flood)
- Domain adaptation for cross-network robustness
- Better edge integration with network telemetry pipelines

Long-term:
- Huawei ecosystem scaling (cloud + edge orchestration)
- Automated mitigation loop with human-in-the-loop controls

---

## Slide 14 — Completeness & Demonstrability (15%)
What we submit for verification:
- Slide deck + report
- Reproducible codebase and scripts
- Saved model checkpoints
- Evaluation logs and figures
- README / reproducibility instructions
- (Optional) open-source proof screenshot for public repo

Judge message:
- Our package is complete, verifiable, and demo-ready

---

## Slide 15 — Why SYNSheild Can Win
- Strong innovation with practical implementation
- High same-domain performance with rigorous reporting
- Huawei-aligned technology stack (MindSpore + mindcv)
- Clear deployment strategy from server to early network-edge detection architecture
- Honest handling of limitations (including no-GPU server constraint) + concrete roadmap

**Thank you — Q&A**

---

## Backup Slide A — Exact Reported Metrics
Use exact numbers from:
- `reports/eval_results_per_second_0p3.md`
- `reports/cross_domain_eval_2026-04-03.md`

---

## Backup Slide B — Commands / Repro
- `bash scripts/run_all_eval_0p3.sh`
- `bash scripts/run_all_paper_figures.sh`

Include environment pins from:
- `reports/REPRODUCIBILITY.md`

