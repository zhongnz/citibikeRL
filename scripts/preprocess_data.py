#!/usr/bin/env python3
"""Preprocess Citi Bike trips into hourly station-level flow counts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from citibikerl.data import missing_required_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Citi Bike raw data.")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output", required=True, help="Path to write processed CSV")
    return parser.parse_args()


def parse_hour(timestamp: str) -> str | None:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(minute=0, second=0, microsecond=0).isoformat()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input does not exist: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Cannot preprocess: CSV header row not found.")
            return 1

        missing = missing_required_columns(reader.fieldnames)
        if missing:
            print("Cannot preprocess: missing required columns:")
            for col in missing:
                print(f" - {col}")
            return 1

        grouped: dict[tuple[str, str, str], int] = defaultdict(int)
        for row in reader:
            hour = parse_hour(row["started_at"])
            if hour is None:
                continue
            key = (hour, row["start_station_id"], row["end_station_id"])
            grouped[key] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "start_station_id", "end_station_id", "trip_count"])
        for (hour, start, end), count in sorted(grouped.items()):
            writer.writerow([hour, start, end, count])

    print(f"Wrote processed dataset: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
