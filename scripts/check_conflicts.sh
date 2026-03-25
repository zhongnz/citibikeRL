#!/usr/bin/env bash
set -euo pipefail

# Search tracked files for git conflict markers.
if rg -n "^(<<<<<<<|=======|>>>>>>>)" --glob '!*.lock' --glob '!.git/**' .; then
  echo "Conflict marker check FAILED. Resolve markers above."
  exit 1
fi

echo "Conflict marker check PASSED."
