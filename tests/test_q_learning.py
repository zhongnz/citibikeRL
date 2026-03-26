"""Tests for the tabular Q-learning loop."""

from __future__ import annotations

import numpy as np

from citibikerl.rebalancing import (
    DemandProfilePolicy,
    ForecastHeuristicPolicy,
    NoOpPolicy,
    QTablePolicy,
    RebalancingEnvConfig,
    TrainingConfig,
    build_q_state_encoder,
    build_demand_profile,
    encode_forecast_state,
    encode_state,
    evaluate_policy,
    train_q_learning,
)
from citibikerl.rebalancing.data import DemandDataset


def test_q_learning_beats_no_op_on_simple_rebalancing_problem() -> None:
    dataset = DemandDataset(
        station_ids=("A", "B"),
        episode_days=("2026-02-01",),
        departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
        arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
    )
    env_config = RebalancingEnvConfig(
        station_capacity=5,
        initial_inventory=1,
        move_amount=3,
        served_reward=1.0,
        unmet_penalty=2.0,
        move_penalty_per_bike=0.0,
        overflow_penalty=0.0,
    )
    training_config = TrainingConfig(
        episodes=200,
        alpha=0.3,
        gamma=0.95,
        epsilon=0.4,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        bucket_size=1,
        seed=3,
    )

    result = train_q_learning(dataset, env_config, training_config)
    baseline_metrics = evaluate_policy(dataset, env_config, NoOpPolicy(), bucket_size=1, policy_name="baseline")
    trained_metrics = evaluate_policy(
        dataset,
        env_config,
        QTablePolicy(result.q_table),
        bucket_size=1,
        policy_name="trained",
        state_encoder=build_q_state_encoder(
            actions=result.actions,
            env_config=env_config,
            training_config=training_config,
            demand_profile=result.demand_profile,
            state_representation=result.state_representation,
        ),
    )

    assert trained_metrics[0]["total_reward"] > baseline_metrics[0]["total_reward"]
    assert trained_metrics[0]["unmet_demand"] < baseline_metrics[0]["unmet_demand"]


def test_q_learning_tracks_guided_exploration_usage() -> None:
    dataset = DemandDataset(
        station_ids=("A", "B"),
        episode_days=("2026-02-01",),
        departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
        arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
    )
    env_config = RebalancingEnvConfig(
        station_capacity=5,
        initial_inventory=1,
        move_amount=3,
        served_reward=1.0,
        unmet_penalty=2.0,
        move_penalty_per_bike=0.0,
        overflow_penalty=0.0,
    )
    training_config = TrainingConfig(
        episodes=1,
        epsilon=1.0,
        epsilon_decay=1.0,
        epsilon_min=1.0,
        heuristic_exploration_bias=1.0,
        bucket_size=1,
        forecast_bucket_size=1,
        seed=3,
    )

    result = train_q_learning(dataset, env_config, training_config)
    metric = result.metrics[0]

    assert metric["exploratory_actions"] == 2
    assert metric["guided_exploration_actions"] == 2
    assert metric["heuristic_match_actions"] == 2


def test_encode_state_includes_calendar_context() -> None:
    state = encode_state(9, 5, 1, (3, 7), bucket_size=2, station_capacity=10)

    assert state == (9, 5, 1, 1, 3)


def test_encode_forecast_state_tracks_profile_action_signal() -> None:
    demand_profile = build_demand_profile(
        DemandDataset(
            station_ids=("A", "B"),
            episode_days=("2026-02-01",),
            departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
            arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
        ),
    )

    state = encode_forecast_state(
        1,
        6,
        1,
        2,
        0,
        11.0,
        0.0,
        0.0,
        6.0,
        (1, 4),
        actions=(None, (0, 1), (1, 0)),
        demand_profile=demand_profile,
        bucket_size=1,
        forecast_bucket_size=1,
        station_capacity=5,
        move_amount=3,
    )

    assert state == (1, 6, 1, 2, 2, 0, 5, 0, 0, 3, 1, 0, 4, 1, 4, 3, 7)


def test_q_table_policy_can_fallback_to_embedded_heuristic_action() -> None:
    state = (5, 1, 0, 7, 2, 4, 3, 2)
    policy = QTablePolicy({}, fallback_policy=ForecastHeuristicPolicy())

    assert policy.select_action(state, 10) == 7


def test_q_table_policy_can_fallback_when_state_has_too_few_visits() -> None:
    state = (5, 1, 0, 7, 2, 4, 3, 2)
    policy = QTablePolicy(
        {state: np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 1.0, 0.0])},
        state_visit_counts={state: 1},
        min_visit_count=2,
        fallback_policy=ForecastHeuristicPolicy(),
    )

    assert policy.select_action(state, 10) == 7


def test_demand_profile_policy_beats_no_op_on_simple_rebalancing_problem() -> None:
    dataset = DemandDataset(
        station_ids=("A", "B"),
        episode_days=("2026-02-01",),
        departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
        arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
    )
    env_config = RebalancingEnvConfig(
        station_capacity=5,
        initial_inventory=1,
        move_amount=3,
        served_reward=1.0,
        unmet_penalty=2.0,
        move_penalty_per_bike=0.0,
        overflow_penalty=0.0,
    )
    training_config = TrainingConfig(episodes=10, bucket_size=1, seed=3)
    demand_profile = build_demand_profile(dataset)

    heuristic_metrics = evaluate_policy(
        dataset,
        env_config,
        DemandProfilePolicy(
            actions=(None, (0, 1), (1, 0)),
            demand_profile=demand_profile,
            bucket_size=training_config.bucket_size,
            station_capacity=env_config.station_capacity,
            move_amount=env_config.move_amount,
        ),
        bucket_size=1,
        policy_name="heuristic",
    )
    baseline_metrics = evaluate_policy(dataset, env_config, NoOpPolicy(), bucket_size=1, policy_name="baseline")

    assert heuristic_metrics[0]["total_reward"] > baseline_metrics[0]["total_reward"]
