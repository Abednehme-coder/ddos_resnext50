# Huawei ICT Competition — Innovation Track (Regional)

**Canonical PDF source:** LaTeX project in `ict_innovation_overleaf/` (Overleaf). This Markdown file is a lightweight mirror and may lag behind `main.tex`.

**Project title:** AI-Based Detection of SYN Flood DDoS Attacks Using ResNeXt-50 and Deep Learning on Packet Images  

**Framework:** Huawei MindSpore + mindcv  

**Team name:** SYNSheild  
**Institution:** Beirut Arab University (BAU), Lebanon  
**Submission date:** 11 April 2026  

**Companion document:** Technical appendix — `reports/ICT_Innovation_Regional_Appendix.md`

---

## Executive summary

Distributed Denial of Service (DDoS) attacks, especially TCP SYN floods, remain a major threat to service availability. This project delivers an end-to-end, reproducible pipeline that turns raw network captures (PCAP) into fixed-size grayscale images and classifies them with a **ResNeXt-50 (32×4d)** model trained in **Huawei MindSpore**. The solution is aligned with the **ICT Innovation** track: it combines a clear real-world problem, a working software pipeline, measurable offline performance on the public **CICDDoS2019** dataset, and optional paths toward deployment on GPU or Huawei-oriented infrastructure.

**Primary benchmark (recommended for judges):** a model trained on **0.3 s time-window** packet images, evaluated on held-out validation and test splits from the same distribution, with additional **blind random sampling** on the test split. On the test split (1845 samples), the system reaches **98.10% accuracy** and **0.9375 macro F1**; a blind test of 270 images (maximum balanced sample from test, seed 42) yields **98.9% accuracy** and **0.9889 macro F1**.

**Legacy configuration (per-packet images):** an earlier training setup achieves very high scores on a large blind test of 3000 images from the per-packet test split (see appendix). **Cross-domain** experiments show that mixing training representation (time-window vs per-packet) without retraining **does not** transfer; operational systems must use the **same** image construction as training. This honesty strengthens the innovation story: the method is strong when the deployment pipeline matches the training pipeline.

---

## 1. Competition alignment (Innovation track)

| Expectation (typical) | How this project addresses it |
|----------------------|-------------------------------|
| Real problem | SYN flood DDoS and availability risk for networks and services |
| Creative / technical solution | Packet-byte → 2D image → ResNeXt; class-balanced training; time-window variant for temporal context |
| Implementation | Scripts for PCAP labeling, conversion, training, evaluation; MindSpore + mindcv |
| Evidence | Quantitative metrics, confusion matrices, blind tests, optional calibration figures |
| Feasibility & honesty | Cross-domain results and deployment notes document limits and next steps |

---

## 2. Problem statement and objectives

**Problem.** Attackers send large volumes of TCP SYN segments to exhaust server resources (half-open connections, state tables, or middleboxes). Operators need automated aids that can flag abusive traffic patterns without relying only on brittle static rules.

**Objectives.**

1. Binary classification: **DDoS (SYN flood)** vs **normal** traffic, starting from PCAP data.  
2. Use a modern deep CNN (**ResNeXt-50**) implemented on **Huawei MindSpore** for training and inference.  
3. Evaluate on **CICDDoS2019** with clear train/validation/test splits and **blind** inference scripts (model sees images only).  
4. Document reproducibility (environment, commands, checkpoints) for regional review.

Non-goals (explicit): full production SOC integration, exhaustive evasion robustness studies, and multi-attack-type detection are out of scope for this submission but noted as future work.

---

## 3. Innovation and differentiation

1. **Architecture.** Prior published work on similar packet-image pipelines used ResNet-50; this project uses **ResNeXt-50 (32×4d)** to exploit grouped convolutions (“cardinality”) for richer feature learning at comparable capacity.  
2. **Huawei stack.** Training and evaluation run on **MindSpore** with **mindcv**, supporting a credible path toward **Ascend** / Huawei Cloud deployment narratives where the competition values ecosystem fit.  
3. **End-to-end automation.** From zipped CICDDoS2019 archives through PCAP-level SYN heuristics, packet-to-image conversion, folder layout, training, and evaluation—documented as a script chain rather than a manual lab notebook.  
4. **Imbalanced security metrics.** **Weighted cross-entropy** from class counts and **macro F1–oriented** checkpoint selection reflect operational interest in both classes, not accuracy alone on skewed data.  
5. **Temporal variant.** A **0.3 s time-window** image representation is trained and evaluated as the **primary** model; legacy **per-packet** results are retained for comparison. Cross-domain evaluation shows that **representation alignment** between train and deploy is mandatory—an important engineering insight for judges.

---

## 4. System overview

**High-level flow.**

1. **Ingest:** PCAP files (e.g. from CICDDoS2019).  
2. **Label (heuristic):** PCAPs with more than a threshold of TCP SYN packets are treated as DDoS for dataset construction; others as normal (details in appendix).  
3. **Convert:** Raw packet bytes are arranged as 2D grayscale and resized to **224×224** (methodology consistent with cited literature).  
4. **Train:** ResNeXt-50, ImageNet-style preprocessing (including normalization), Adam / AdamWeightDecay, weighted loss.  
5. **Evaluate:** Held-out val/test metrics, blind random draws from test, optional ROC/PR/calibration scripts.

**Primary artifact for regional judging.**

- **Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`  
- **Data root:** `dataset/images_per_second_window_0p3/` (train / val / test under the same 0.3 s construction)

**Suggested figure for slides (later):** one diagram with five boxes: PCAP → Label → Image builder → ResNeXt → Decision.

---

## 5. Dataset and methodology (summary)

**Dataset.** **CICDDoS2019** (Canadian Institute for Cybersecurity)—public, widely used for DDoS research, containing realistic benign and attack traffic including SYN flood scenarios.

**Splits.** Train, validation, and test image folders are used as in the repository layout (`train/`, `val/`, `test/` with `ddos/` and `normal/` subfolders). Class indices follow **alphabetical** folder order (`ddos` = 0, `normal` = 1) consistently across scripts.

**Model.** ResNeXt-50 32×4d; input **224×224** RGB where grayscale is replicated to three channels; **two-class** softmax output.

**Training highlights (primary 0.3 s run).**

- Optimizer: Adam / AdamWeightDecay; learning rate **1e-3**, weight decay **1e-4** (see training script for exact flags used in your run).  
- Batch size **16**, **8** epochs in documented configuration.  
- **WeightedCrossEntropyLoss** for class imbalance.  
- Best checkpoint selected by **macro F1** on validation.

Full hyperparameter tables and file paths are in the appendix.

---

## 6. Results (main tables)

### 6.1 Primary model — 0.3 s time-window images

**Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`  
**Evaluation date (recorded):** 2026-04-03 (GPU; see appendix for environment pin)

| Split | Samples | Accuracy | Macro F1 |
|------|---------|----------|----------|
| Validation | 1842 | 98.21% | 0.9399 |
| Test | 1845 | 98.10% | 0.9375 |

**Test confusion matrix** [rows = true class, columns = predicted]:

|  | pred ddos | pred normal |
|--|----------:|------------:|
| true ddos | 135 | 0 |
| true normal | 35 | 1675 |

**Blind test** (`blind_test_random.py`, test split, seed 42, **270** images = max equal per class): **267/270 correct (98.9%)**, macro F1 **0.9889**.

### 6.2 Legacy model — per-packet images (reference)

For historical comparison, a **per-packet** image dataset under `dataset/images/` was trained to approximately **95.56%** test macro F1. A **large blind test** of **3000** images (1500 per class, shuffled) reported **99.9%** accuracy and **0.9993** macro F1. These numbers apply **only** to the per-packet representation and checkpoint documented in the appendix; they are **not** interchangeable with the 0.3 s primary model without matching preprocessing.

### 6.3 Calibration note (optional depth)

Scripts can produce ROC/PR and calibration diagnostics (e.g. test ECE on the order of **0.074** in one logged run, with a heuristic threshold near **0.87** for max F1—see `reports/paper_figures_2026-04-03.log`). Use the appendix for reproduction commands.

---

## 7. Limitations, risks, and ethics

1. **Dataset scope.** Metrics reflect **CICDDoS2019** and the team’s labeling heuristics; real networks may differ (protocol mix, encryption, sampling point).  
2. **Representation alignment.** Cross-domain evaluation shows **poor transfer** if test images are built differently from training (e.g. 0.3 s model on per-packet folders). Production pipelines must mirror training.  
3. **Heuristic labels.** PCAP-level SYN counts are a **proxy** label for dataset construction, not ground truth from an operator.  
4. **Security and privacy.** PCAPs can contain sensitive payloads; handle data under institutional policy; do not distribute captures in the submission without permission.  
5. **Adversarial and evasion** scenarios are not evaluated here.

---

## 8. Demo and reproducibility (for judges)

**Minimal story.** “We convert PCAPs to images, train ResNeXt on MindSpore, and report confusion matrices and F1 on held-out test data plus a blind sample.”

**Reproduce primary metrics (after dataset and checkpoint are in place):**

```bash
cd /path/to/huawei
bash scripts/run_all_eval_0p3.sh
```

Details, alternative single commands, Python version, MindSpore version, and `pip freeze` reference are in **`reports/ICT_Innovation_Regional_Appendix.md`** and **`reports/REPRODUCIBILITY.md`**.

**Optional:** If the team operates a live demo service, document access method **without** exposing credentials in a public repo; prefer screen captures or judge-only handout.

---

## 9. Future work

- Align **live capture windows** (e.g. `run_pipeline.py` / service windowing) strictly with the **0.3 s** training representation, or retrain on the live representation.  
- Extend to **multi-class** DDoS families or encrypted traffic (different features).  
- Quantify **latency and throughput** on target hardware (GPU / Ascend).  
- Integrate with **Huawei Cloud** or edge deployment templates if required by later competition stages.

---

## 10. References and resources

- CICDDoS2019 — Canadian Institute for Cybersecurity.  
- Bazzi et al., *ResNet-Based Detection of SYN Flood DDoS Attacks* (methodological basis for packet-to-image representation).  
- Huawei MindSpore — https://www.mindspore.cn/  
- mindcv — MindSpore Computer Vision toolkit.

---

## 11. Team and contributions

- **Wafik Ibrahim** — PCAP pipeline, dataset construction, labeling scripts  
- **Mohammed Farhat** — MindSpore training, evaluation, checkpoints  
- **Abed Al Rida Nehme** — Reporting, reproducibility, deployment integration  

**Instructor:** Ali Nassar

---

*End of main report. See `reports/ICT_Innovation_Regional_Appendix.md` for extended tables, script index, cross-domain study, and reproducibility snapshot.*
