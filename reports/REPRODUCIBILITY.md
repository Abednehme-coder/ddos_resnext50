# Reproducibility snapshot

Captured on **2026-04-03** from the development machine used for GPU evaluation.

| Item | Value |
|------|--------|
| Git commit | `96b82d4cf152053175950a346bdd204b6e109ebf` |
| Python | 3.10.12 |
| MindSpore | 2.6.0 |
| mindcv | 0.3.0 |
| GPU | NVIDIA GeForce RTX 3070 |
| NVIDIA driver | 535.288.01 |
| CUDA toolkit (eval scripts) | `/usr/local/cuda-11.6` (typical MindSpore GPU build) |

## Commands

- Same-domain offline metrics: `bash scripts/run_all_eval_0p3.sh` (defaults to `DEVICE=GPU`).
- ROC/PR/calibration figures: `bash scripts/run_all_paper_figures.sh` → `research/figures/*.pdf`.
- Cross-domain: see `reports/cross_domain_eval_2026-04-03.md`.

Full `pip freeze` (optional):

```bash
.venv/bin/pip freeze > reports/pip_freeze_2026-04-03.txt
```
