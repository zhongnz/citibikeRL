#!/usr/bin/env python3
"""Run the full tabular rebalancing experiment workflow."""

from __future__ import annotations

import argparse

from citibikerl.config import load_yaml_section
from citibikerl.rebalancing import (
    RebalancingEnvConfig,
    TrainingConfig,
    build_output_paths,
    build_station_activity_summary,
    load_demand_dataset,
    normalize_station_ids,
    run_experiment,
)


def first_not_none(*values: object) -> object:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end rebalancing experiment workflow.")
    parser.add_argument("--input", required=True, help="Processed hourly flows CSV or comma-separated CSVs")
    parser.add_argument("--weather-input", help="Optional NOAA daily weather CSV aligned by day")
    parser.add_argument("--output-prefix", required=True, help="Prefix used to name experiment artifacts")
    parser.add_argument(
        "--environment-config",
        default="configs/environment.yaml",
        help="YAML file with environment settings",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training.yaml",
        help="YAML file with training settings",
    )
    parser.add_argument("--station-ids", help="Comma-separated station IDs to use")
    parser.add_argument("--top-n-stations", type=int, help="Fallback station count when station IDs are not provided")
    parser.add_argument("--station-capacity", type=int, help="Maximum bikes held at each station")
    parser.add_argument("--initial-inventory", type=int, help="Initial bike inventory per station")
    parser.add_argument("--move-amount", type=int, help="Transfer size per action")
    parser.add_argument("--served-reward", type=float, help="Reward for a served trip")
    parser.add_argument("--unmet-penalty", type=float, help="Penalty for an unmet trip")
    parser.add_argument("--move-penalty-per-bike", type=float, help="Penalty per bike moved")
    parser.add_argument("--overflow-penalty", type=float, help="Penalty per overflowed bike")
    parser.add_argument("--episodes", type=int, help="Number of Q-learning episodes")
    parser.add_argument("--train-fraction", type=float, help="Chronological share of days reserved for training")
    parser.add_argument("--test-start-day", help="Explicit YYYY-MM-DD boundary for the held-out test split")
    parser.add_argument("--alpha", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, help="Per-episode exploration decay")
    parser.add_argument("--epsilon-min", type=float, help="Minimum exploration rate")
    parser.add_argument(
        "--heuristic-exploration-bias",
        type=float,
        help="Share of exploratory actions that follow the demand-profile heuristic",
    )
    parser.add_argument(
        "--min-state-visit-count",
        type=int,
        help="Minimum training visit count required before trusting the learned Q action",
    )
    parser.add_argument("--bucket-size", type=int, help="Inventory bucket size for state discretization")
    parser.add_argument(
        "--forecast-bucket-size",
        type=int,
        help="Bucket size for demand-aware Q-state summary features",
    )
    parser.add_argument("--seed", type=int, help="RNG seed")
    parser.add_argument(
        "--state-representation",
        help="Q-table state representation tag (forecast_profile_v1..v4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment_values = load_yaml_section(args.environment_config, "environment")
    training_values = load_yaml_section(args.training_config, "training")

    station_ids = normalize_station_ids(args.station_ids or training_values.get("station_ids"))
    top_n_stations = int(first_not_none(args.top_n_stations, training_values.get("top_n_stations"), 5))

    dataset = load_demand_dataset(
        args.input,
        station_ids=station_ids,
        top_n_stations=top_n_stations,
        weather_input=args.weather_input,
    )
    station_summary = build_station_activity_summary(
        args.input,
        station_ids=dataset.station_ids,
        top_n_stations=dataset.num_stations,
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
    training_config = TrainingConfig(
        train_fraction=float(first_not_none(args.train_fraction, training_values.get("train_fraction"), 0.75)),
        test_start_day=str(first_not_none(args.test_start_day, training_values.get("test_start_day"), "")).strip() or None,
        episodes=int(first_not_none(args.episodes, training_values.get("episodes"), 300)),
        alpha=float(first_not_none(args.alpha, training_values.get("alpha"), 0.2)),
        gamma=float(first_not_none(args.gamma, training_values.get("gamma"), 0.95)),
        epsilon=float(first_not_none(args.epsilon, training_values.get("epsilon"), 0.35)),
        epsilon_decay=float(first_not_none(args.epsilon_decay, training_values.get("epsilon_decay"), 0.995)),
        epsilon_min=float(first_not_none(args.epsilon_min, training_values.get("epsilon_min"), 0.05)),
        heuristic_exploration_bias=float(
            first_not_none(
                args.heuristic_exploration_bias,
                training_values.get("heuristic_exploration_bias"),
                0.0,
            ),
        ),
        min_state_visit_count=int(
            first_not_none(args.min_state_visit_count, training_values.get("min_state_visit_count"), 1),
        ),
        bucket_size=int(first_not_none(args.bucket_size, training_values.get("bucket_size"), 2)),
        forecast_bucket_size=int(
            first_not_none(args.forecast_bucket_size, training_values.get("forecast_bucket_size"), 2),
        ),
        seed=int(first_not_none(args.seed, training_values.get("seed"), 7)),
        state_representation=str(
            first_not_none(
                args.state_representation,
                training_values.get("state_representation"),
                "forecast_profile_v4",
            ),
        ),
    )

    summary = run_experiment(
        input_path=args.input,
        weather_input=args.weather_input,
        dataset=dataset,
        station_summary=station_summary,
        env_config=env_config,
        training_config=training_config,
        output_paths=build_output_paths(args.output_prefix),
    )

    print(f"Ran experiment '{args.output_prefix}' with {summary['station_count']} station(s).")
    print(f"Selected stations: {', '.join(summary['selected_station_ids'])}")
    print(
        "{split} baseline avg reward={baseline:.2f}; heuristic avg reward={heuristic:.2f}; q+fallback avg reward={trained:.2f}".format(
            split=summary["primary_eval_split"],
            baseline=summary["baseline_summary"]["avg_reward"],
            heuristic=summary["heuristic_summary"]["avg_reward"],
            trained=summary["trained_summary"]["avg_reward"],
        ),
    )
    print(
        "Saved q+fallback avg reward={saved:.2f}; model written to {model_path}".format(
            saved=summary["saved_policy_summary"]["avg_reward"],
            model_path=summary["outputs"]["model"],
        ),
    )
    print(f"Summary JSON: {summary['outputs']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
