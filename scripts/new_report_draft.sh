#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v1}"
[[ -z "$VERSION" ]] && VERSION="v1"
OUT="docs/report/report_draft_${VERSION}.md"
TEMPLATE="docs/report/report_outline.md"

if [[ -e "$OUT" ]]; then
  echo "Report draft already exists: $OUT"
  exit 0
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template missing: $TEMPLATE"
  exit 1
fi

cp "$TEMPLATE" "$OUT"

echo "Created: $OUT"
