"""Dataset schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from citibikerl.config import load_yaml_section

DEFAULT_REQUIRED_COLUMNS = (
    "started_at",
    "ended_at",
    "start_station_id",
    "start_station_name",
    "end_station_id",
    "end_station_name",
)
DEFAULT_TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class DatasetSettings:
    """Runtime settings that control raw-data validation and preprocessing."""

    required_columns: tuple[str, ...] = DEFAULT_REQUIRED_COLUMNS
    timezone: str = DEFAULT_TIMEZONE


def load_dataset_settings(path: str | Path | None = "configs/dataset.yaml") -> DatasetSettings:
    """Load dataset settings from YAML, falling back to repo defaults."""
    values = load_yaml_section(path, "dataset")
    required_columns = values.get("required_columns", DEFAULT_REQUIRED_COLUMNS)
    timezone = values.get("timezone", DEFAULT_TIMEZONE)

    if not isinstance(required_columns, (list, tuple)) or not all(
        isinstance(column, str) and column.strip() for column in required_columns
    ):
        raise ValueError("Dataset config 'required_columns' must be a non-empty list of column names.")
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("Dataset config 'timezone' must be a non-empty string.")

    return DatasetSettings(
        required_columns=tuple(column.strip() for column in required_columns),
        timezone=timezone.strip(),
    )


REQUIRED_COLUMNS = list(DEFAULT_REQUIRED_COLUMNS)
