#!/usr/bin/env python3
"""Evaluate a previously saved Q-learning policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from citibikerl.rebalancing import (
    DemandProfilePolicy,
    ForecastHeuristicPolicy,
    NoOpPolicy,
    QTablePolicy,
    RebalancingEnv,
    build_q_state_encoder,
    evaluate_policy,
    load_demand_dataset,
    load_model,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Q-learning policy.")
    parser.add_argument("--input", required=True, help="Processed hourly flows CSV or comma-separated CSVs")
    parser.add_argument("--weather-input", help="Optional NOAA daily weather CSV aligned by day")
    parser.add_argument("--model", required=True, help="Saved model JSON from train_q_learning.py")
    parser.add_argument("--output", required=True, help="Output CSV for per-episode metrics")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only evaluate the saved policy and omit the no-op baseline rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    saved_model = load_model(args.model)
    dataset = load_demand_dataset(
        args.input,
        station_ids=saved_model.station_ids,
        top_n_stations=len(saved_model.station_ids),
        weather_input=args.weather_input,
    )

    env = RebalancingEnv(dataset, saved_model.env_config)
    if env.actions != saved_model.actions:
        raise ValueError("Saved model action space does not match the reconstructed environment.")
    q_state_encoder = build_q_state_encoder(
        actions=saved_model.actions,
        env_config=saved_model.env_config,
        training_config=saved_model.training_config,
        demand_profile=saved_model.demand_profile,
        state_representation=saved_model.state_representation,
    )
    forecast_fallback_policy = ForecastHeuristicPolicy()

    metrics_frames = []
    if not args.skip_baseline:
        baseline_metrics = evaluate_policy(
            dataset,
            saved_model.env_config,
            NoOpPolicy(),
            bucket_size=saved_model.training_config.bucket_size,
            policy_name="baseline_no_op",
        )
        metrics_frames.append(pd.DataFrame(baseline_metrics))
        if saved_model.demand_profile is not None:
            heuristic_metrics = evaluate_policy(
                dataset,
                saved_model.env_config,
                DemandProfilePolicy(
                    actions=saved_model.actions,
                    demand_profile=saved_model.demand_profile,
                    bucket_size=saved_model.training_config.bucket_size,
                    station_capacity=saved_model.env_config.station_capacity,
                    move_amount=saved_model.env_config.move_amount,
                ),
                bucket_size=saved_model.training_config.bucket_size,
                policy_name="heuristic_demand_profile",
            )
            metrics_frames.append(pd.DataFrame(heuristic_metrics))

    trained_metrics = evaluate_policy(
        dataset,
        saved_model.env_config,
        QTablePolicy(
            saved_model.q_table,
            state_visit_counts=saved_model.state_visit_counts,
            min_visit_count=saved_model.training_config.min_state_visit_count,
            fallback_policy=forecast_fallback_policy,
        ),
        bucket_size=saved_model.training_config.bucket_size,
        policy_name="saved_q_policy_with_heuristic_fallback",
        state_encoder=q_state_encoder,
    )
    metrics_frames.append(pd.DataFrame(trained_metrics))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame = pd.concat(metrics_frames, ignore_index=True)
    output_frame.to_csv(output_path, index=False)

    trained_summary = summarize_metrics(trained_metrics)
    print(f"Evaluated saved policy across {dataset.num_episodes} episode(s) and {dataset.num_stations} station(s).")
    print(f"Selected stations: {', '.join(dataset.station_ids)}")
    print(
        "Saved policy avg reward={avg_reward:.2f}, served={avg_served_trips:.2f}, unmet={avg_unmet_demand:.2f}".format(
            **trained_summary,
        ),
    )
    print(f"Wrote metrics to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
