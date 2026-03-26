#!/usr/bin/env python3
"""Evaluate the no-op baseline on processed Citi Bike hourly flows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from citibikerl.config import load_yaml_section
from citibikerl.rebalancing import (
    NoOpPolicy,
    RebalancingEnvConfig,
    evaluate_policy,
    load_demand_dataset,
    normalize_station_ids,
    summarize_metrics,
)


def first_not_none(*values: object) -> object:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the no-op baseline policy.")
    parser.add_argument("--input", required=True, help="Processed hourly flows CSV or comma-separated CSVs")
    parser.add_argument("--weather-input", help="Optional NOAA daily weather CSV aligned by day")
    parser.add_argument("--output", required=True, help="Output CSV for per-episode metrics")
    parser.add_argument(
        "--environment-config",
        default="configs/environment.yaml",
        help="YAML file with environment settings",
    )
    parser.add_argument(
        "--evaluation-config",
        default="configs/evaluation.yaml",
        help="YAML file with evaluation settings",
    )
    parser.add_argument(
        "--station-ids",
        help="Comma-separated station IDs to use instead of top-N activity selection",
    )
    parser.add_argument("--top-n-stations", type=int, help="Fallback station count when station IDs are not provided")
    parser.add_argument("--station-capacity", type=int, help="Maximum bikes held at each station")
    parser.add_argument("--initial-inventory", type=int, help="Initial bike inventory per station")
    parser.add_argument("--move-amount", type=int, help="Transfer size per action")
    parser.add_argument("--served-reward", type=float, help="Reward for a served trip")
    parser.add_argument("--unmet-penalty", type=float, help="Penalty for an unmet trip")
    parser.add_argument("--move-penalty-per-bike", type=float, help="Penalty per bike moved")
    parser.add_argument("--overflow-penalty", type=float, help="Penalty per overflowed bike")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment_values = load_yaml_section(args.environment_config, "environment")
    evaluation_values = load_yaml_section(args.evaluation_config, "evaluation")

    station_ids = normalize_station_ids(args.station_ids or evaluation_values.get("station_ids"))
    top_n_stations = int(first_not_none(args.top_n_stations, evaluation_values.get("top_n_stations"), 5))

    dataset = load_demand_dataset(
        args.input,
        station_ids=station_ids,
        top_n_stations=top_n_stations,
        weather_input=args.weather_input,
    )
    env_config = RebalancingEnvConfig(
        station_capacity=int(first_not_none(args.station_capacity, environment_values.get("station_capacity"), 20)),
        initial_inventory=int(
            first_not_none(args.initial_inventory, environment_values.get("initial_inventory"), 10),
        ),
        move_amount=int(first_not_none(args.move_amount, environment_values.get("move_amount"), 3)),
        served_reward=float(first_not_none(args.served_reward, environment_values.get("served_reward"), 1.0)),
        unmet_penalty=float(first_not_none(args.unmet_penalty, environment_values.get("unmet_penalty"), 2.0)),
        move_penalty_per_bike=float(
            first_not_none(args.move_penalty_per_bike, environment_values.get("move_penalty_per_bike"), 0.05),
        ),
        overflow_penalty=float(
            first_not_none(args.overflow_penalty, environment_values.get("overflow_penalty"), 0.5),
        ),
    )

    metrics = evaluate_policy(dataset, env_config, NoOpPolicy(), policy_name="baseline_no_op")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_path, index=False)

    summary = summarize_metrics(metrics)
    print(f"Evaluated baseline across {dataset.num_episodes} episode(s) and {dataset.num_stations} station(s).")
    print(f"Selected stations: {', '.join(dataset.station_ids)}")
    print(
        "Average reward={avg_reward:.2f}, served={avg_served_trips:.2f}, unmet={avg_unmet_demand:.2f}".format(
            **summary,
        ),
    )
    print(f"Wrote metrics to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
