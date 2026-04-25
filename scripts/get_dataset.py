#!/usr/bin/env python3
"""Download Citi Bike dataset file into data/raw and emit metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import zipfile


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
    saved_path = download_dataset(args.url, output_path)
    print(f"Saved dataset to: {saved_path}")

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


def download_dataset(url: str, output_path: Path) -> Path:
    """Download a source file, extracting ZIP-backed CSVs when requested."""
    if _url_path(url).lower().endswith(".zip") and output_path.suffix.lower() != ".zip":
        archive_path = output_path.with_suffix(output_path.suffix + ".download")
        urlretrieve(url, archive_path)
        try:
            if zipfile.is_zipfile(archive_path):
                extract_first_csv(archive_path, output_path)
            else:
                archive_path.replace(output_path)
        finally:
            if archive_path.exists():
                archive_path.unlink()
        return output_path

    urlretrieve(url, output_path)
    return output_path


def extract_first_csv(archive_path: Path, output_path: Path) -> None:
    """Extract the first CSV member from a ZIP archive."""
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]
        if not csv_members:
            raise ValueError(f"Downloaded ZIP archive does not contain a CSV file: {archive_path}")

        with archive.open(csv_members[0], "r") as source, output_path.open("wb") as destination:
            destination.write(source.read())


def _url_path(url: str) -> str:
    return urlparse(url).path or url


if __name__ == "__main__":
    raise SystemExit(main())
