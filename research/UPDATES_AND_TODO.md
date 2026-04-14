# Research project — updates so far & remaining work

This file summarizes what has been implemented in the repository and what is still open. Paths are relative to the repo root unless noted.

---

## Completed updates

### Report (LaTeX)

- **`research/main.tex`** — Full report structure: abstract, introduction, methods (pipeline, ResNeXt, loss, MindSpore), results (primary 0.3 s metrics, confusion tables, per-class metrics, blind test, legacy vs 0.3 s comparison, deployment notes, limitations, conclusion), appendices (threat model, notation, extended related work, FAQ, reproducibility, optional generated figures).
- **Cross-domain table removed** — The confusing cross-domain argmax table was removed from the PDF narrative; train/serve gap is still described qualitatively (e.g. offline window images vs live per-packet images).
- **`research/acronyms.tex`**, **`research/references.bib`** — Supporting files for the report.
- **`research/COMPETITION_SLIDE.md`** — Short slide outline for the competition.

### Primary model & data (0.3 s time windows)

- **Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (ResNeXt-50, focal + DDoS-focused early stopping).
- **Dataset:** `dataset/images_per_second_window_0p3/` (train/val/test from the same 0.3 s distribution).
- **Defaults** across scripts and `run_pipeline.py` point to this checkpoint and 0.3 s windows where applicable.

### Evaluation & testing

- **`scripts/run_all_eval_0p3.sh`** — One-shot val/test + blind (270) on GPU by default.
- **`scripts/eval_test_detailed.py`**, **`scripts/blind_test_random.py`**, **`scripts/eval_roc_pr_calibration.py`** — Metrics, blind sampling, ROC/PR/ECE; defaults use the primary 0.3 s dataset and checkpoint.
- **`scripts/eval_primary_ckpt.py`** — Policy: standard eval scripts only accept the primary focal-v2 checkpoint unless `HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1` (used e.g. by `eval_per_second_1s_model.sh`).
- **`scripts/eval_cross_domain.py`** — Still in repo for optional domain-shift experiments; **not** featured in the main report table anymore.
- **`reports/eval_results_per_second_0p3.md`** — Recorded offline metrics and reproduce commands.
- **`reports/REPRODUCIBILITY.md`**, **`reports/pip_freeze_2026-04-03.txt`** — Environment snapshot.
- **`reports/paper_figures_2026-04-03.log`**, **`research/figures/*.pdf`** — ROC/PR/threshold/ECE figures when generated.
- **`reports/cross_domain_eval_2026-04-03.md`** — Log of optional cross-domain GPU runs (kept for traceability, not in main PDF table).
- **`tests/`** — Includes `test_eval_primary_ckpt.py` and `test_metrics_from_log.py`; **`pytest`** added to **`requirements.txt`**.

### Live deployment (ECS)

- **`depployemnt/systemd/synflood-detector.service`** — `--window-sec 0.3`, `--ckpt` → `.../model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`.
- **`depployemnt/deploy_ckpt_to_ecs.sh`** — Uploads the primary checkpoint to the same path on the server.
- **`depployemnt/install_ecs_systemd.sh`** — Installs/refreshes the unit on ECS.
- **`depployemnt/OPERATIONS.md`**, **`depployemnt/PROGRESS.md`** — Operations notes and progress (including ECS checkpoint + systemd alignment when last run).

### Pipeline code

- **`run_pipeline.py`** — Live/offline pipeline; default checkpoint and window length aligned with 0.3 s training; alert fields (`streak`, `alert`, thresholds) as implemented in repo.

---

## Still to do (or verify)

### Build the PDF

- **`pdflatex`** (and BibTeX/biber if you use bibliography) was not available on the original dev host without a full TeX install.
- **Action:** Build `research/main.tex` on a machine with TeX (local install, CI, or Overleaf). Fix any compile warnings if they appear.
- **Glossaries:** `main.tex` uses **`\makenoidxglossaries`** so abbreviations resolve with **pdflatex only** (no separate `makeglossaries` run).
- **Bibliography:** With natbib + `\bibliography{references}`, run **BibTeX** on the same job name as pdflatex, e.g. `pdflatex main` → `bibtex main` → `pdflatex` ×2. If you use **`-jobname=output`**, run **`bibtex output`** (not `bibtex main`), or citations stay undefined.

### Optional polish

- **References:** Tighten any placeholder entries in `references.bib` (e.g. exact metadata for related work you cite).
- **Slides:** Refresh **`research/COMPETITION_SLIDE.md`** if numbers or wording changed after LaTeX edits.
- **1 s vs 0.3 s comparison:** If the jury needs it, run the existing per-second scripts and add one line or a small table to slides (not required for the core story).

### Operations (ongoing)

- **Live traffic:** Tune **`--alert-min-p-ddos`** / **`--alert-consecutive`** from real `detections.log` if public Internet noise produces too many `pred: ddos` lines.
- **Train/serve gap:** Live **`run_pipeline.py`** uses per-packet images in each window; offline training uses window-aggregated images — monitor behavior; no single offline number proves production accuracy without labeled captures.

### Repository hygiene (optional)

- Ensure **ECS** still matches repo after future edits: re-run **`deploy_ckpt_to_ecs.sh`** and **`install_ecs_systemd.sh`** when checkpoint or unit files change.
- After **`pip install -r requirements.txt`**, run **`pytest tests/`** locally to match CI.

---

*Last updated to reflect removal of the cross-domain table from `main.tex` and current deployment defaults.*
