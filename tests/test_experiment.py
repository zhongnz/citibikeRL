"""Tests for the end-to-end experiment workflow helper."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from citibikerl.rebalancing import (
    RebalancingEnvConfig,
    TrainingConfig,
    build_output_paths,
    build_station_activity_summary,
    load_demand_dataset,
    run_experiment,
)


def test_run_experiment_writes_all_artifacts(tmp_path: Path) -> None:
    processed_csv = tmp_path / "hourly_flows.csv"
    processed_csv.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T00:00:00-05:00,X,B,3",
                "2026-02-01T01:00:00-05:00,A,X,4",
                "2026-02-02T00:00:00-05:00,X,B,3",
                "2026-02-02T01:00:00-05:00,A,X,4",
            ],
        ),
        encoding="utf-8",
    )

    dataset = load_demand_dataset(processed_csv, station_ids=["A", "B"])
    station_summary = build_station_activity_summary(processed_csv, station_ids=dataset.station_ids)
    outputs = build_output_paths("test_run", outputs_root=tmp_path / "outputs")
    summary = run_experiment(
        input_path=processed_csv,
        weather_input=None,
        dataset=dataset,
        station_summary=station_summary,
        env_config=RebalancingEnvConfig(
            station_capacity=5,
            initial_inventory=1,
            move_amount=3,
            move_penalty_per_bike=0.0,
            overflow_penalty=0.0,
        ),
        training_config=TrainingConfig(
            episodes=30,
            bucket_size=1,
            epsilon=0.4,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            seed=3,
        ),
        output_paths=outputs,
    )

    assert outputs.model_path.exists()
    assert outputs.training_metrics_path.exists()
    assert outputs.evaluation_metrics_path.exists()
    assert outputs.saved_policy_metrics_path.exists()
    assert outputs.selected_stations_path.exists()
    assert outputs.reward_plot_path.exists()
    assert outputs.comparison_plot_path.exists()
    assert outputs.summary_path.exists()

    summary_payload = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["selected_station_ids"] == ["A", "B"]
    assert summary_payload["primary_eval_split"] == "test"
    assert summary_payload["train_episode_count"] == 1
    assert summary_payload["test_episode_count"] == 1
    assert summary["trained_summary"]["avg_reward"] >= summary["baseline_summary"]["avg_reward"]

    selected_station_frame = pd.read_csv(outputs.selected_stations_path)
    assert selected_station_frame["station_id"].tolist() == ["A", "B"]
    evaluation_frame = pd.read_csv(outputs.evaluation_metrics_path)
    assert set(evaluation_frame["split"]) == {"train", "test"}


def test_run_experiment_supports_explicit_day_boundary_split(tmp_path: Path) -> None:
    processed_csv_jan = tmp_path / "hourly_flows_jan.csv"
    processed_csv_feb = tmp_path / "hourly_flows_feb.csv"
    processed_csv_jan.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-01-31T00:00:00-05:00,X,B,3",
                "2026-01-31T01:00:00-05:00,A,X,4",
            ],
        ),
        encoding="utf-8",
    )
    processed_csv_feb.write_text(
        "\n".join(
            [
                "hour,start_station_id,end_station_id,trip_count",
                "2026-02-01T00:00:00-05:00,X,B,3",
                "2026-02-01T01:00:00-05:00,A,X,4",
            ],
        ),
        encoding="utf-8",
    )

    combined_input = f"{processed_csv_jan},{processed_csv_feb}"
    dataset = load_demand_dataset(combined_input, station_ids=["A", "B"])
    station_summary = build_station_activity_summary(combined_input, station_ids=dataset.station_ids)
    outputs = build_output_paths("test_run_boundary", outputs_root=tmp_path / "outputs")
    summary = run_experiment(
        input_path=combined_input,
        weather_input=None,
        dataset=dataset,
        station_summary=station_summary,
        env_config=RebalancingEnvConfig(
            station_capacity=5,
            initial_inventory=1,
            move_amount=3,
            move_penalty_per_bike=0.0,
            overflow_penalty=0.0,
        ),
        training_config=TrainingConfig(
            episodes=30,
            test_start_day="2026-02-01",
            bucket_size=1,
            epsilon=0.4,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            seed=3,
        ),
        output_paths=outputs,
    )

    assert summary["train_episode_count"] == 1
    assert summary["test_episode_count"] == 1
    assert summary["split_strategy"] == "explicit_day_boundary"
