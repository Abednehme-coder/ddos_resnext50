# ICT competition — one-slide story (outline)

**Title:** SYN flood detection with ResNeXt-50 on packet / window images (MindSpore)

**Problem:** DDoS floods exhaust services; need scalable detection beyond static rules.

**Idea:** Turn traffic into images → classify with ResNeXt-50; train on CICDDoS2019; primary model uses **0.3 s time windows** (`per_second_0p3_focal_v2`).

**Numbers (offline, same domain):** ~**98.1%** test accuracy, macro F1 ~**0.94**; blind test **98.9%** on max balanced sample.

**Live demo:** ECS runs `run_pipeline.py`; JSON logs include `p_ddos`, `streak`, **`alert`** (webhook only when threshold + consecutive windows met). **Many `pred: ddos` lines ≠ confirmed attack** — Internet noise / train–serve gap; use thresholds for real alerts.

**Stack:** Huawei MindSpore, mindcv, optional CPU inference on cloud.

**Takeaway:** Strong **lab** metrics + honest **operational** framing; deployment tuning (`--alert-min-p-ddos`, `--alert-consecutive`) reduces alert spam.
