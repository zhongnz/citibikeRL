#!/usr/bin/env python3
"""Validate raw dataset schema against required columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from citibikerl.data import missing_required_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Citi Bike raw dataset schema.")
    parser.add_argument("--input", required=True, help="Raw CSV input path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Input dataset not found: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Dataset schema invalid: header row not found.")
            return 1
        missing = missing_required_columns(reader.fieldnames)

    if missing:
        print("Dataset schema invalid. Missing columns:")
        for col in missing:
            print(f" - {col}")
        return 1

    print("Dataset schema validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
