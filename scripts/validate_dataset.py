#!/usr/bin/env python3
"""Validate raw dataset schema against required columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from citibikerl.data import load_dataset_settings, missing_required_columns, open_csv_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Citi Bike raw dataset schema.")
    parser.add_argument("--input", required=True, help="Raw CSV input path")
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="YAML file with dataset settings such as required columns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    dataset_settings = load_dataset_settings(args.dataset_config)

    if not input_path.exists():
        print(f"Input dataset not found: {input_path}")
        return 1

    with open_csv_text(input_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Dataset schema invalid: header row not found.")
            return 1
        missing = missing_required_columns(reader.fieldnames, dataset_settings.required_columns)

    if missing:
        print("Dataset schema invalid. Missing columns:")
        for col in missing:
            print(f" - {col}")
        return 1

    print("Dataset schema validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
