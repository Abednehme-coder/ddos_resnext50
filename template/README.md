# Huawei ICT Innovation — Regional template

## Files in this folder

| File | Role |
|------|------|
| `[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage).pdf` | **Official blank template (PDF)** — use it to check slide order, margins, and what the jury expects visually. Do not submit this empty file as your final deck. |
| `[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage) .pptx` | **Editable source** — fill your content here, then export to PDF for submission (or submit PPTX if the portal asks for it). |

Note: the `.pptx` name has a **space** before `.pptx`; the `.pdf` has **no** space before `.pdf`.

## PPTX → PDF (after you edit the slides)

**On this machine (with LibreOffice):**

```bash
chmod +x scripts/convert_innovation_template_to_pdf.sh
scripts/convert_innovation_template_to_pdf.sh
```

Install if needed: `sudo apt install libreoffice-impress`

**Without LibreOffice:** open `[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage) .pptx` in **Microsoft PowerPoint** or **Google Slides** → **Export / Download as PDF**.

The script writes: `template/Template2_Innovation_Regional_Slides.pdf` (sanitized name).

Close the presentation before converting (remove `.~lock.*` files if LibreOffice complains).

## Before submitting slides

Rename the file to match the competition naming rule, e.g.:

`[Template 2] Innovation Competition Presentation Slides - <Area> - SYNSheild (Regional Stage).pptx`

Fill **Area** per your track and replace **Team Name** with **SYNSheild** (or your official spelling).

## Regional stage — submission checklist (summary)

Align your package with what the jury expects:

| Item | Your project |
|------|----------------|
| **Template 2 presentation** | Fill template; export PPTX/PDF as required |
| **Code & data** | Repo: training (`notebook/train_resnext.py`), inference/eval scripts, **partial** dataset sample or instructions, **saved `.ckpt`**, training/eval **logs** |
| **README** | Reproduce: env, `bash scripts/run_all_eval_0p3.sh`, data layout |
| **Open-source proof** | Screenshot of GitHub/Gitee **public** repo (if you open-source) |
| **Consistency** | Slide numbers must match **code + logs** (committee may verify) |

**Scoring (typical weights):** Innovation 40%, Application value 35%, Completeness & demonstrability 15% (includes dataset description, key code, performance verification, demo), Presentation & Q&A 10%.

Written report PDF: use `ict_innovation_overleaf/` (separate from this slide template).
