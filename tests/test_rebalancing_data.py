"""Tests for processed demand dataset loading."""

from __future__ import annotations

from pathlib import Path

from citibikerl.rebalancing import (
    build_station_activity_summary,
    load_demand_dataset,
    split_demand_dataset_by_day,
    split_demand_dataset_temporal,
)


def test_load_demand_dataset_builds_daily_tensors(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T09:00:00-05:00,101,201,3",
                "2026-02-01T09:00:00-05:00,201,101,2",
                "2026-02-02T10:00:00-05:00,101,201,4",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, station_ids=["101", "201"])

    assert dataset.num_episodes == 2
    assert dataset.horizon == 24
    assert dataset.station_ids == ("101", "201")
    assert dataset.departures[0, 9].tolist() == [3.0, 2.0]
    assert dataset.arrivals[0, 9].tolist() == [2.0, 3.0]
    assert dataset.departures[1, 10].tolist() == [4.0, 0.0]
    assert dataset.daily_context is not None
    assert dataset.daily_context[0].day_of_week == 6
    assert dataset.daily_context[1].day_of_week == 0


def test_load_demand_dataset_selects_top_activity_stations(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T09:00:00-05:00,101,201,10",
                "2026-02-01T10:00:00-05:00,102,202,2",
                "2026-02-01T11:00:00-05:00,103,203,1",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, top_n_stations=2)

    assert dataset.station_ids == ("101", "201")


def test_load_demand_dataset_accepts_multiple_processed_inputs(tmp_path: Path) -> None:
    processed_csv_a = tmp_path / "hourly_flows_a.csv"
    processed_csv_b = tmp_path / "hourly_flows_b.csv"
    processed_csv_a.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-01-31T23:00:00-05:00,101,201,2",
            ],
        ),
        encoding="utf-8",
    )
    processed_csv_b.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T00:00:00-05:00,101,201,3",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(f"{processed_csv_a},{processed_csv_b}", station_ids=["101", "201"])

    assert dataset.episode_days == ("2026-01-31", "2026-02-01")
    assert dataset.departures[0, 23].tolist() == [2.0, 0.0]
    assert dataset.departures[1, 0].tolist() == [3.0, 0.0]


def test_load_demand_dataset_filters_standardized_monthly_files_to_their_own_month(tmp_path: Path) -> None:
    january_processed = tmp_path / "jc_202601_hourly_flows.csv"
    february_processed = tmp_path / "jc_202602_hourly_flows.csv"
    january_processed.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-01-31T23:00:00-05:00,101,201,2",
            ],
        ),
        encoding="utf-8",
    )
    february_processed.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-01-31T23:00:00-05:00,101,201,99",
                "2026-02-01T00:00:00-05:00,101,201,3",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(f"{january_processed},{february_processed}", station_ids=["101", "201"])

    assert dataset.episode_days == ("2026-01-31", "2026-02-01")
    assert dataset.departures[0, 23].tolist() == [2.0, 0.0]
    assert dataset.departures[1, 0].tolist() == [3.0, 0.0]


def test_build_station_activity_summary_matches_selected_stations(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T09:00:00-05:00,101,201,3",
                "2026-02-01T10:00:00-05:00,201,101,2",
                "2026-02-01T11:00:00-05:00,101,301,4",
            ],
        ),
        encoding="utf-8",
    )

    summary = build_station_activity_summary(processed_csv, station_ids=["101", "201"])

    assert summary["station_id"].tolist() == ["101", "201"]
    assert summary["total_activity"].tolist() == [9.0, 5.0]


def test_split_demand_dataset_temporal_preserves_chronological_order(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T09:00:00-05:00,101,201,1",
                "2026-02-02T09:00:00-05:00,101,201,1",
                "2026-02-03T09:00:00-05:00,101,201,1",
                "2026-02-04T09:00:00-05:00,101,201,1",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, station_ids=["101", "201"])
    split = split_demand_dataset_temporal(dataset, 0.5)

    assert split.train_dataset.episode_days == ("2026-02-01", "2026-02-02")
    assert split.test_dataset is not None
    assert split.test_dataset.episode_days == ("2026-02-03", "2026-02-04")


def test_split_demand_dataset_by_day_uses_explicit_boundary(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-01-31T09:00:00-05:00,101,201,1",
                "2026-02-01T09:00:00-05:00,101,201,1",
                "2026-02-02T09:00:00-05:00,101,201,1",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, station_ids=["101", "201"])
    split = split_demand_dataset_by_day(dataset, "2026-02-01")

    assert split.train_dataset.episode_days == ("2026-01-31",)
    assert split.test_dataset is not None
    assert split.test_dataset.episode_days == ("2026-02-01", "2026-02-02")


def test_load_demand_dataset_merges_weather_context_by_day(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    weather_csv = tmp_path / "weather.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2025-01-01T09:00:00-05:00,101,201,3",
                "2025-01-02T09:00:00-05:00,101,201,4",
            ],
        ),
        encoding="utf-8",
    )
    weather_csv.write_text(
        "\n".join(
            [
                "DATE,TAVG,PRCP,SNOW,AWND",
                "2025-01-01,8.6,0.0,0.0,6.7",
                "2025-01-02,4.6,1.5,0.0,8.8",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, station_ids=["101", "201"], weather_input=weather_csv)

    assert dataset.daily_context is not None
    assert dataset.daily_context[0].is_holiday == 1
    assert dataset.daily_context[0].temperature_c == 8.6
    assert dataset.daily_context[1].precipitation_mm == 1.5
    assert dataset.daily_context[1].wind_speed_m_s == 8.8
