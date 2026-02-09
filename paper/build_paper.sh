#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${ROOT_DIR}/paper"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
BUILD_DIR="${PAPER_DIR}/build"

mkdir -p "${BUILD_DIR}"

python3 "${PAPER_DIR}/build_paper_assets.py" \
  --outputs-dir "${OUTPUTS_DIR}" \
  --paper-dir "${PAPER_DIR}"

cd "${PAPER_DIR}"
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory="build" \
  "main.tex"

cp "${BUILD_DIR}/main.pdf" "WHSDSC_2026_paper.pdf"
echo "Paper built: ${PAPER_DIR}/WHSDSC_2026_paper.pdf"
