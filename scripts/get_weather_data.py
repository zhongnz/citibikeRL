#!/usr/bin/env python3
"""Download normalized NOAA daily weather summaries for experiment context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


DEFAULT_DATA_TYPES = ("TAVG", "TMAX", "TMIN", "PRCP", "SNOW", "AWND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA daily weather summaries.")
    parser.add_argument("--station", required=True, help="NOAA station id, e.g. USW00014734")
    parser.add_argument("--start-date", required=True, help="Start day in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End day in YYYY-MM-DD format")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--data-types",
        default=",".join(DEFAULT_DATA_TYPES),
        help="Comma-separated NOAA data types to request",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_types = [part.strip().upper() for part in args.data_types.split(",") if part.strip()]
    url = build_request_url(
        station=args.station,
        start_date=args.start_date,
        end_date=args.end_date,
        data_types=data_types,
    )
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)

    frame = normalize_noaa_daily_summaries(pd.DataFrame(payload))
    if frame.empty:
        raise ValueError(f"No weather rows returned for station={args.station} between {args.start_date} and {args.end_date}.")
    frame.to_csv(output_path, index=False)

    metadata = {
        "source_url": url,
        "station": args.station,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "row_count": int(len(frame)),
        "columns": frame.columns.tolist(),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote weather dataset: {output_path}")
    print(f"Wrote weather metadata: {metadata_path}")
    return 0


def build_request_url(*, station: str, start_date: str, end_date: str, data_types: list[str]) -> str:
    query = urlencode(
        {
            "dataset": "daily-summaries",
            "stations": station,
            "startDate": start_date,
            "endDate": end_date,
            "dataTypes": ",".join(data_types),
            "includeAttributes": "false",
            "units": "metric",
            "format": "json",
        },
    )
    return f"https://www.ncei.noaa.gov/access/services/data/v1?{query}"


def normalize_noaa_daily_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "DATE": "day",
        "STATION": "station_id",
        "TAVG": "temperature_c",
        "TMAX": "temperature_max_c",
        "TMIN": "temperature_min_c",
        "PRCP": "precipitation_mm",
        "SNOW": "snowfall_mm",
        "AWND": "wind_speed_m_s",
    }
    normalized = frame.rename(columns=rename_map).copy()
    required_columns = ["day", "station_id", "temperature_c", "precipitation_mm", "snowfall_mm", "wind_speed_m_s"]
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"NOAA response is missing columns: {', '.join(missing_columns)}")

    for column in normalized.columns:
        if column == "day" or column == "station_id":
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["day"] = pd.to_datetime(normalized["day"], errors="raise").dt.strftime("%Y-%m-%d")
    normalized["station_id"] = normalized["station_id"].astype(str)
    ordered_columns = [
        "day",
        "station_id",
        "temperature_c",
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_mm",
        "snowfall_mm",
        "wind_speed_m_s",
    ]
    return normalized[ordered_columns].sort_values("day", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
