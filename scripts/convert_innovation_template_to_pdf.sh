#!/usr/bin/env bash
# Convert Huawei ICT Innovation regional PPTX template to PDF.
# Requires: LibreOffice (soffice or libreoffice). Install: sudo apt install libreoffice-impress
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${ROOT}/template"
PPTX="${TEMPLATE_DIR}/[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage) .pptx"
OUT_NAME="Template2_Innovation_Regional_Slides.pdf"
OUT="${TEMPLATE_DIR}/${OUT_NAME}"

if [[ ! -f "${PPTX}" ]]; then
  echo "Missing: ${PPTX}" >&2
  exit 1
fi

convert_with() {
  local cmd="$1"
  shift
  "${cmd}" --headless --convert-to pdf --outdir "${TEMPLATE_DIR}" "$@" "${PPTX}"
}

if command -v libreoffice >/dev/null 2>&1; then
  convert_with libreoffice
elif command -v soffice >/dev/null 2>&1; then
  convert_with soffice
else
  echo "LibreOffice not found. Install with:" >&2
  echo "  sudo apt install libreoffice-impress" >&2
  echo "Or open the .pptx in PowerPoint / Google Slides and use Export / Print to PDF." >&2
  exit 1
fi

# LibreOffice names the PDF from the .pptx basename (brackets in filename).
GEN="${TEMPLATE_DIR}/[Template 2] Innovation Competition Presentation Slides - Area - Team Name (Regional Stage) .pdf"
if [[ -f "${GEN}" ]]; then
  mv -f "${GEN}" "${OUT}"
  echo "Wrote: ${OUT}"
else
  # Some versions drop brackets in output name
  shopt -s nullglob
  for f in "${TEMPLATE_DIR}/"*.pdf; do
    if [[ "${f}" -nt "${PPTX}" ]]; then
      mv -f "${f}" "${OUT}"
      echo "Wrote: ${OUT}"
      exit 0
    fi
  done
  echo "Conversion ran but PDF not found next to template. Check: ${TEMPLATE_DIR}/" >&2
  exit 1
fi
