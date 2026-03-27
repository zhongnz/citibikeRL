"""Tests for dataset validation helpers."""

from pathlib import Path

from citibikerl.data import load_dataset_settings, missing_required_columns


def test_missing_required_columns_detects_missing() -> None:
    cols = ["started_at", "ended_at", "start_station_id"]
    missing = missing_required_columns(cols)
    assert "end_station_id" in missing
    assert "start_station_name" in missing


def test_missing_required_columns_passes_when_complete() -> None:
    cols = [
        "started_at",
        "ended_at",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
    ]
    assert missing_required_columns(cols) == []


def test_missing_required_columns_uses_custom_required_columns() -> None:
    cols = ["started_at", "ended_at", "bike_type"]
    assert missing_required_columns(cols, required_columns=["started_at", "bike_type"]) == []


def test_load_dataset_settings_reads_yaml_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "dataset:\n"
        "  timezone: UTC\n"
        "  required_columns:\n"
        "    - started_at\n"
        "    - bike_type\n",
        encoding="utf-8",
    )

    settings = load_dataset_settings(config_path)

    assert settings.timezone == "UTC"
    assert settings.required_columns == ("started_at", "bike_type")
