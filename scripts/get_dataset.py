#!/usr/bin/env python3
"""Download Citi Bike dataset file into data/raw and emit metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from urllib.request import urlretrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a dataset file into data/raw.")
    parser.add_argument("--url", required=True, help="Dataset URL")
    parser.add_argument("--output", required=True, help="Output file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from: {args.url}")
    urlretrieve(args.url, output_path)
    print(f"Saved dataset to: {output_path}")

    meta = {
        "url": args.url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "path": str(output_path),
    }
    sidecar_metadata_path = Path(f"{output_path}.metadata.json")
    sidecar_metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metadata_path = output_path.parent / "_dataset_metadata.json"
    existing_records: list[dict[str, str]] = []
    if metadata_path.exists():
        existing_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(existing_payload, list):
            existing_records = [record for record in existing_payload if isinstance(record, dict)]
        elif isinstance(existing_payload, dict):
            existing_records = [existing_payload]

    updated_records = [record for record in existing_records if record.get("path") != str(output_path)]
    updated_records.append(meta)
    metadata_path.write_text(json.dumps(updated_records, indent=2), encoding="utf-8")
    print(f"Wrote metadata sidecar: {sidecar_metadata_path}")
    print(f"Updated metadata index: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
