#!/usr/bin/env bash
set -euo pipefail

DATE_STR="${1:-$(date +%F)}"
[[ -z "$DATE_STR" ]] && DATE_STR="$(date +%F)"
OUT="docs/notes/meeting_${DATE_STR}.md"
TEMPLATE="docs/notes/meeting_YYYY-MM-DD_template.md"

if [[ -e "$OUT" ]]; then
  echo "Meeting note already exists: $OUT"
  exit 0
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template missing: $TEMPLATE"
  exit 1
fi

cp "$TEMPLATE" "$OUT"

echo "Created: $OUT"
