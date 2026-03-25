"""Tests for dataset validation helpers."""

from citibikerl.data import missing_required_columns


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
