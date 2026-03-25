#!/usr/bin/env bash
set -euo pipefail

required_paths=(
  "README.md"
  "pyproject.toml"
  ".gitignore"
  "data/raw"
  "data/processed"
  "docs/proposal/proposal_outline.md"
  "docs/report/report_outline.md"
  "docs/presentation/slide_outline.md"
  "references/datasets/CITIBIKE_DATA_SOURCE.md"
  "outputs/figures"
  "scripts"
  "scripts/preprocess_data.py"
  "data/raw/README.md"
  "configs/dataset.yaml"
  "scripts/validate_dataset.py"
  "scripts/get_dataset.py"
  "src/citibikerl"
  "src/citibikerl/__init__.py"
  "tests"
)

missing=()
for p in "${required_paths[@]}"; do
  if [[ ! -e "$p" ]]; then
    missing+=("$p")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Repository structure check FAILED. Missing paths:"
  for m in "${missing[@]}"; do
    echo "  - $m"
  done
  exit 1
fi

echo "Repository structure check PASSED."
