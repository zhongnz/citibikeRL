#!/usr/bin/env python3
"""Train and evaluate a NumPy DQN agent for bike rebalancing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from citibikerl.config import load_yaml_section
from citibikerl.rebalancing import (
    DQNPolicy,
    DQNTrainingConfig,
    DemandProfilePolicy,
    NoOpPolicy,
    RebalancingEnvConfig,
    build_dense_state_encoder,
    evaluate_dqn_policy,
    evaluate_policy,
    load_demand_dataset,
    normalize_station_ids,
    save_dqn_model,
    split_demand_dataset_by_day,
    split_demand_dataset_temporal,
    summarize_metrics,
    train_dqn,
)


def first_not_none(*values: object) -> object:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dueling double DQN policy.")
    parser.add_argument("--input", required=True, help="Processed hourly flows CSV or comma-separated CSVs")
    parser.add_argument("--weather-input", help="Optional NOAA daily weather CSV aligned by day")
    parser.add_argument("--output-model", required=True, help="Output JSON path for learned DQN weights")
    parser.add_argument("--output-training-metrics", required=True, help="Output CSV path for per-training-episode metrics")
    parser.add_argument("--output-evaluation-metrics", required=True, help="Output CSV path for evaluation metrics")
    parser.add_argument("--environment-config", default="configs/environment.yaml", help="YAML file with environment settings")
    parser.add_argument("--training-config", default="configs/dqn_training.yaml", help="YAML file with DQN training settings")
    parser.add_argument("--station-ids", help="Comma-separated station IDs to use")
    parser.add_argument("--top-n-stations", type=int, help="Fallback station count when station IDs are not provided")
    parser.add_argument("--station-capacity", type=int, help="Maximum bikes held at each station")
    parser.add_argument("--initial-inventory", type=int, help="Initial bike inventory per station")
    parser.add_argument("--move-amount", type=int, help="Transfer size per action")
    parser.add_argument("--served-reward", type=float, help="Reward for a served trip")
    parser.add_argument("--unmet-penalty", type=float, help="Penalty for an unmet trip")
    parser.add_argument("--move-penalty-per-bike", type=float, help="Penalty per bike moved")
    parser.add_argument("--overflow-penalty", type=float, help="Penalty per overflowed bike")
    parser.add_argument("--episodes", type=int, help="Number of DQN training episodes")
    parser.add_argument("--train-fraction", type=float, help="Chronological share of days reserved for training")
    parser.add_argument("--test-start-day", help="Explicit YYYY-MM-DD boundary for the held-out test split")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, help="Per-episode exploration decay")
    parser.add_argument("--epsilon-min", type=float, help="Minimum exploration rate")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, help="Replay minibatch size")
    parser.add_argument("--replay-capacity", type=int, help="Replay buffer capacity")
    parser.add_argument("--replay-warmup", type=int, help="Replay size before training starts")
    parser.add_argument("--hidden-dim", type=int, help="Hidden layer width")
    parser.add_argument("--heuristic-bucket-size", type=int, help="Inventory bucket size used by heuristic guidance/baseline")
    parser.add_argument("--target-update-interval", type=int, help="Hard target update period in env steps")
    parser.add_argument("--train-interval", type=int, help="Gradient update frequency in env steps")
    parser.add_argument("--gradient-clip", type=float, help="Global gradient norm clip")
    parser.add_argument("--heuristic-exploration-bias", type=float, help="Share of exploratory actions that follow the heuristic")
    parser.add_argument("--move-action-margin", type=float, help="Evaluation-time Q-value margin over no_op before taking a move action")
    parser.add_argument("--double-dqn", dest="double_dqn", action="store_true", help="Enable Double DQN target selection")
    parser.add_argument("--no-double-dqn", dest="double_dqn", action="store_false", help="Disable Double DQN target selection")
    parser.add_argument("--dueling", dest="dueling", action="store_true", help="Enable dueling value/advantage heads")
    parser.add_argument("--no-dueling", dest="dueling", action="store_false", help="Disable dueling value/advantage heads")
    parser.add_argument("--seed", type=int, help="RNG seed")
    parser.set_defaults(double_dqn=None, dueling=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment_values = load_yaml_section(args.environment_config, "environment")
    training_values = load_yaml_section(args.training_config, "dqn_training")

    station_ids = normalize_station_ids(args.station_ids or training_values.get("station_ids"))
    top_n_stations = int(first_not_none(args.top_n_stations, training_values.get("top_n_stations"), 5))

    dataset = load_demand_dataset(
        args.input,
        station_ids=station_ids,
        top_n_stations=top_n_stations,
        weather_input=args.weather_input,
    )
    env_config = RebalancingEnvConfig(
        station_capacity=int(first_not_none(args.station_capacity, environment_values.get("station_capacity"), 20)),
        initial_inventory=int(first_not_none(args.initial_inventory, environment_values.get("initial_inventory"), 10)),
        move_amount=int(first_not_none(args.move_amount, environment_values.get("move_amount"), 3)),
        served_reward=float(first_not_none(args.served_reward, environment_values.get("served_reward"), 1.0)),
        unmet_penalty=float(first_not_none(args.unmet_penalty, environment_values.get("unmet_penalty"), 2.0)),
        move_penalty_per_bike=float(first_not_none(args.move_penalty_per_bike, environment_values.get("move_penalty_per_bike"), 0.05)),
        overflow_penalty=float(first_not_none(args.overflow_penalty, environment_values.get("overflow_penalty"), 0.5)),
    )
    training_config = DQNTrainingConfig(
        train_fraction=float(first_not_none(args.train_fraction, training_values.get("train_fraction"), 0.75)),
        test_start_day=str(first_not_none(args.test_start_day, training_values.get("test_start_day"), "")).strip() or None,
        episodes=int(first_not_none(args.episodes, training_values.get("episodes"), 400)),
        gamma=float(first_not_none(args.gamma, training_values.get("gamma"), 0.99)),
        epsilon=float(first_not_none(args.epsilon, training_values.get("epsilon"), 0.35)),
        epsilon_decay=float(first_not_none(args.epsilon_decay, training_values.get("epsilon_decay"), 0.995)),
        epsilon_min=float(first_not_none(args.epsilon_min, training_values.get("epsilon_min"), 0.05)),
        learning_rate=float(first_not_none(args.learning_rate, training_values.get("learning_rate"), 1e-3)),
        batch_size=int(first_not_none(args.batch_size, training_values.get("batch_size"), 64)),
        replay_capacity=int(first_not_none(args.replay_capacity, training_values.get("replay_capacity"), 10000)),
        replay_warmup=int(first_not_none(args.replay_warmup, training_values.get("replay_warmup"), 256)),
        hidden_dim=int(first_not_none(args.hidden_dim, training_values.get("hidden_dim"), 64)),
        heuristic_bucket_size=int(first_not_none(args.heuristic_bucket_size, training_values.get("heuristic_bucket_size"), 2)),
        target_update_interval=int(first_not_none(args.target_update_interval, training_values.get("target_update_interval"), 100)),
        train_interval=int(first_not_none(args.train_interval, training_values.get("train_interval"), 1)),
        gradient_clip=float(first_not_none(args.gradient_clip, training_values.get("gradient_clip"), 5.0)),
        heuristic_exploration_bias=float(first_not_none(args.heuristic_exploration_bias, training_values.get("heuristic_exploration_bias"), 0.25)),
        move_action_margin=float(first_not_none(args.move_action_margin, training_values.get("move_action_margin"), 0.0)),
        double_dqn=bool(first_not_none(args.double_dqn, training_values.get("double_dqn"), True)),
        dueling=bool(first_not_none(args.dueling, training_values.get("dueling"), True)),
        seed=int(first_not_none(args.seed, training_values.get("seed"), 7)),
    )

    dataset_split = (
        split_demand_dataset_by_day(dataset, training_config.test_start_day)
        if training_config.test_start_day
        else split_demand_dataset_temporal(dataset, training_config.train_fraction)
    )
    train_dataset = dataset_split.train_dataset
    test_dataset = dataset_split.test_dataset
    result = train_dqn(train_dataset, env_config, training_config)
    state_encoder = build_dense_state_encoder(
        demand_profile=result.demand_profile,
        station_capacity=env_config.station_capacity,
    )
    trained_policy = DQNPolicy(
        network_state=result.network_state,
        state_encoder=state_encoder,
        dueling=training_config.dueling,
        move_action_margin=training_config.move_action_margin,
    )

    model_path = Path(args.output_model)
    training_metrics_path = Path(args.output_training_metrics)
    evaluation_metrics_path = Path(args.output_evaluation_metrics)
    for path in (model_path, training_metrics_path, evaluation_metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    save_dqn_model(
        model_path,
        station_ids=dataset.station_ids,
        actions=result.actions,
        network_state=result.network_state,
        env_config=env_config,
        training_config=training_config,
        demand_profile=result.demand_profile,
        state_representation=result.state_representation,
        feature_dim=result.feature_dim,
    )
    pd.DataFrame(result.metrics).to_csv(training_metrics_path, index=False)

    evaluation_frames: list[pd.DataFrame] = []
    split_summaries: dict[str, dict[str, float]] = {}
    for split_name, split_dataset in [("train", train_dataset)] + ([] if test_dataset is None else [("test", test_dataset)]):
        baseline_metrics = [
            {**metric, "split": split_name}
            for metric in evaluate_policy(split_dataset, env_config, NoOpPolicy(), policy_name="baseline_no_op")
        ]
        heuristic_metrics = [
            {**metric, "split": split_name}
            for metric in evaluate_policy(
                split_dataset,
                env_config,
                DemandProfilePolicy(
                    actions=result.actions,
                    demand_profile=result.demand_profile,
                    bucket_size=training_config.heuristic_bucket_size,
                    station_capacity=env_config.station_capacity,
                    move_amount=env_config.move_amount,
                ),
                bucket_size=training_config.heuristic_bucket_size,
                policy_name="heuristic_demand_profile",
            )
        ]
        trained_metrics = [
            {**metric, "split": split_name}
            for metric in evaluate_dqn_policy(
                split_dataset,
                env_config,
                trained_policy,
                policy_name="trained_dqn_policy",
            )
        ]
        evaluation_frames.extend(
            [pd.DataFrame(baseline_metrics), pd.DataFrame(heuristic_metrics), pd.DataFrame(trained_metrics)],
        )
        split_summaries[split_name] = {
            "baseline_reward": summarize_metrics(baseline_metrics)["avg_reward"],
            "heuristic_reward": summarize_metrics(heuristic_metrics)["avg_reward"],
            "trained_reward": summarize_metrics(trained_metrics)["avg_reward"],
            "baseline_unmet": summarize_metrics(baseline_metrics)["avg_unmet_demand"],
            "heuristic_unmet": summarize_metrics(heuristic_metrics)["avg_unmet_demand"],
            "trained_unmet": summarize_metrics(trained_metrics)["avg_unmet_demand"],
        }

    pd.concat(evaluation_frames, ignore_index=True).to_csv(evaluation_metrics_path, index=False)

    primary_split = "test" if test_dataset is not None else "train"
    primary_summary = split_summaries[primary_split]
    print(
        f"Trained on {train_dataset.num_episodes} train episode(s), "
        f"evaluated on {0 if test_dataset is None else test_dataset.num_episodes} held-out test episode(s), "
        f"{dataset.num_stations} station(s), {training_config.episodes} DQN episode(s)."
    )
    print(f"Selected stations: {', '.join(dataset.station_ids)}")
    print(
        "{split} baseline avg reward={avg_reward:.2f}; heuristic avg reward={heuristic_reward:.2f}; trained avg reward={trained_reward:.2f}".format(
            split=primary_split,
            avg_reward=primary_summary["baseline_reward"],
            heuristic_reward=primary_summary["heuristic_reward"],
            trained_reward=primary_summary["trained_reward"],
        ),
    )
    print(
        "{split} baseline avg unmet={avg_unmet_demand:.2f}; heuristic avg unmet={heuristic_unmet:.2f}; trained avg unmet={trained_unmet:.2f}".format(
            split=primary_split,
            avg_unmet_demand=primary_summary["baseline_unmet"],
            heuristic_unmet=primary_summary["heuristic_unmet"],
            trained_unmet=primary_summary["trained_unmet"],
        ),
    )
    print(f"Wrote model to: {model_path}")
    print(f"Wrote training metrics to: {training_metrics_path}")
    print(f"Wrote evaluation metrics to: {evaluation_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
