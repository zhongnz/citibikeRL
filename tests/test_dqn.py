"""Tests for the NumPy DQN training path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from citibikerl.rebalancing import (
    DQNPolicy,
    DQNTrainingConfig,
    RebalancingEnvConfig,
    build_dense_state_encoder,
    evaluate_dqn_policy,
    load_dqn_model,
    save_dqn_model,
    train_dqn,
)
from citibikerl.rebalancing.data import DemandDataset


def build_simple_dataset() -> DemandDataset:
    return DemandDataset(
        station_ids=("A", "B"),
        episode_days=("2026-02-01",),
        departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
        arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
    )


def build_simple_env_config() -> RebalancingEnvConfig:
    return RebalancingEnvConfig(
        station_capacity=5,
        initial_inventory=1,
        move_amount=3,
        served_reward=1.0,
        unmet_penalty=2.0,
        move_penalty_per_bike=0.0,
        overflow_penalty=0.0,
    )


def build_simple_training_config() -> DQNTrainingConfig:
    return DQNTrainingConfig(
        episodes=250,
        replay_capacity=2000,
        replay_warmup=32,
        batch_size=32,
        hidden_dim=32,
        target_update_interval=20,
        heuristic_exploration_bias=0.5,
        epsilon=0.4,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        learning_rate=0.005,
        seed=3,
    )


def test_dqn_beats_no_op_on_simple_rebalancing_problem() -> None:
    dataset = build_simple_dataset()
    env_config = build_simple_env_config()
    training_config = build_simple_training_config()

    result = train_dqn(dataset, env_config, training_config)
    trained_policy = DQNPolicy(
        result.network_state,
        build_dense_state_encoder(
            demand_profile=result.demand_profile,
            station_capacity=env_config.station_capacity,
        ),
        dueling=training_config.dueling,
    )

    trained_metrics = evaluate_dqn_policy(dataset, env_config, trained_policy)

    assert trained_metrics[0]["total_reward"] == 4.0
    assert trained_metrics[0]["unmet_demand"] == 0.0


def test_dqn_policy_move_action_margin_falls_back_to_no_op() -> None:
    dataset = build_simple_dataset()
    env_config = build_simple_env_config()
    training_config = build_simple_training_config()
    result = train_dqn(dataset, env_config, training_config)
    state_encoder = build_dense_state_encoder(
        demand_profile=result.demand_profile,
        station_capacity=env_config.station_capacity,
    )

    aggressive_policy = DQNPolicy(
        result.network_state,
        state_encoder,
        dueling=training_config.dueling,
        move_action_margin=0.0,
    )
    gated_policy = DQNPolicy(
        result.network_state,
        state_encoder,
        dueling=training_config.dueling,
        move_action_margin=1e9,
    )

    aggressive_metrics = evaluate_dqn_policy(dataset, env_config, aggressive_policy)
    gated_metrics = evaluate_dqn_policy(dataset, env_config, gated_policy)

    assert aggressive_metrics[0]["moved_bikes"] >= 1
    assert gated_metrics[0]["moved_bikes"] == 0


def test_save_and_load_dqn_model_round_trip(tmp_path: Path) -> None:
    dataset = build_simple_dataset()
    env_config = build_simple_env_config()
    training_config = build_simple_training_config()
    result = train_dqn(dataset, env_config, training_config)
    model_path = tmp_path / "dqn_model.json"

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

    saved_model = load_dqn_model(model_path)
    policy = DQNPolicy(
        saved_model.network_state,
        build_dense_state_encoder(
            demand_profile=saved_model.demand_profile,
            station_capacity=saved_model.env_config.station_capacity,
        ),
        dueling=saved_model.training_config.dueling,
    )
    metrics = evaluate_dqn_policy(dataset, saved_model.env_config, policy)

    assert saved_model.station_ids == ("A", "B")
    assert saved_model.actions == result.actions
    assert saved_model.feature_dim == result.feature_dim
    assert metrics[0]["total_reward"] == 4.0
