"""Daily calendar and weather context aligned to demand episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


@dataclass(frozen=True)
class DailyContext:
    """Per-day exogenous features exposed to the environment."""

    day_of_week: int
    is_weekend: int
    month_of_year: int
    is_holiday: int = 0
    temperature_c: float = 0.0
    precipitation_mm: float = 0.0
    snowfall_mm: float = 0.0
    wind_speed_m_s: float = 0.0


def build_daily_context(
    day_labels: list[str] | tuple[str, ...],
    *,
    weather_input: str | Path | None = None,
) -> tuple[DailyContext, ...]:
    """Build aligned daily calendar/weather context rows for each episode day."""
    day_frame = pd.DataFrame({"day": [str(day_label) for day_label in day_labels]})
    if day_frame.empty:
        return ()

    day_frame["date"] = pd.to_datetime(day_frame["day"], errors="raise")
    day_frame["day_of_week"] = day_frame["date"].dt.dayofweek.astype(int)
    day_frame["is_weekend"] = (day_frame["day_of_week"] >= 5).astype(int)
    day_frame["month_of_year"] = day_frame["date"].dt.month.astype(int)

    holiday_index = USFederalHolidayCalendar().holidays(
        start=day_frame["date"].min(),
        end=day_frame["date"].max(),
    )
    holiday_days = set(pd.Series(holiday_index).dt.strftime("%Y-%m-%d"))
    day_frame["is_holiday"] = day_frame["day"].isin(holiday_days).astype(int)

    if weather_input is not None:
        weather_frame = load_weather_context_frame(weather_input)
        day_frame = day_frame.merge(weather_frame, on="day", how="left", sort=False)
    else:
        day_frame["temperature_c"] = 0.0
        day_frame["precipitation_mm"] = 0.0
        day_frame["snowfall_mm"] = 0.0
        day_frame["wind_speed_m_s"] = 0.0

    day_frame["temperature_c"] = pd.to_numeric(day_frame["temperature_c"], errors="coerce")
    day_frame["wind_speed_m_s"] = pd.to_numeric(day_frame["wind_speed_m_s"], errors="coerce")
    for column in ("precipitation_mm", "snowfall_mm"):
        day_frame[column] = pd.to_numeric(day_frame[column], errors="coerce").fillna(0.0)

    temperature_median = float(day_frame["temperature_c"].dropna().median()) if day_frame["temperature_c"].notna().any() else 0.0
    wind_median = float(day_frame["wind_speed_m_s"].dropna().median()) if day_frame["wind_speed_m_s"].notna().any() else 0.0
    day_frame["temperature_c"] = day_frame["temperature_c"].fillna(temperature_median)
    day_frame["wind_speed_m_s"] = day_frame["wind_speed_m_s"].fillna(wind_median)

    return tuple(
        DailyContext(
            day_of_week=int(row.day_of_week),
            is_weekend=int(row.is_weekend),
            month_of_year=int(row.month_of_year),
            is_holiday=int(row.is_holiday),
            temperature_c=float(row.temperature_c),
            precipitation_mm=float(row.precipitation_mm),
            snowfall_mm=float(row.snowfall_mm),
            wind_speed_m_s=float(row.wind_speed_m_s),
        )
        for row in day_frame.itertuples(index=False)
    )


def load_weather_context_frame(input_path: str | Path) -> pd.DataFrame:
    """Load a NOAA daily weather export and normalize its column names."""
    weather_path = Path(input_path)
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather input file not found: {weather_path}")

    frame = pd.read_csv(weather_path)
    if frame.empty:
        raise ValueError(f"Weather input file is empty: {weather_path}")

    normalized = _normalize_weather_columns(frame)
    grouped = (
        normalized.groupby("day", sort=True)[
            ["temperature_c", "precipitation_mm", "snowfall_mm", "wind_speed_m_s"]
        ]
        .mean()
        .reset_index()
    )
    return grouped


def _normalize_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    rename_map = {
        "DATE": "day",
        "TAVG": "temperature_c",
        "PRCP": "precipitation_mm",
        "SNOW": "snowfall_mm",
        "AWND": "wind_speed_m_s",
    }
    normalized = normalized.rename(columns=rename_map)

    required_columns = ["day", "temperature_c", "precipitation_mm", "snowfall_mm", "wind_speed_m_s"]
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"Weather input file is missing columns: {', '.join(missing_columns)}")

    normalized = normalized[required_columns].copy()
    normalized["day"] = pd.to_datetime(normalized["day"], errors="raise").dt.strftime("%Y-%m-%d")
    return normalized


def summarize_weather_context(weather_input: str | Path) -> dict[str, Any]:
    """Provide a compact summary of a weather input file for experiment metadata."""
    frame = pd.read_csv(weather_input)
    normalized = _normalize_weather_columns(frame)
    return {
        "input_path": str(Path(weather_input)),
        "day_count": int(normalized["day"].nunique()),
        "start_day": str(normalized["day"].min()),
        "end_day": str(normalized["day"].max()),
    }
