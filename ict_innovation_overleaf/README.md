# ICT Innovation report (Overleaf)

Standalone LaTeX project for the Huawei **ICT Innovation (regional)** report. **Not** the BAU thesis (`research/main.tex`).

## Files

| File / folder | Role |
|---------------|------|
| `main.tex` | Full narrative: related work, baselines, TikZ pipeline, optional figures, deployment note, team macros |
| `appendix.tex` | Technical appendix |
| `references.bib` | Bibliography (includes related-work entries) |
| `figures/` | Optional: `sample_ddos.png`, `sample_normal.png`, `roc_pr_test.pdf` (see `figures/README.txt`) |

## Figures

Until you add images, the PDF shows **placeholder boxes**. Export two $224{\times}224$ samples from your dataset and copy an ROC/PR PDF from `research/figures/` after running `bash scripts/run_all_paper_figures.sh` locally.

## Overleaf

1. Zip: `zip -r ict_innovation_overleaf.zip ict_innovation_overleaf/`
2. Upload project; main document = **`main.tex`**
3. **pdfLaTeX** + BibTeX (default)

## Edit before submission

In `main.tex` preamble, set **member names and roles**: `\MemberOneName`, `\MemberTwoName`, `\MemberThreeName`, `\AdvisorName` (or remove the advisor line in the source if not allowed).

Verify **`references.bib`** entry `ictcompetition` URL matches your competition cycle.

## Markdown mirror

Older Markdown drafts: `reports/ICT_Innovation_Regional_Report.md`. The **canonical** write-up for submission is this LaTeX project.
