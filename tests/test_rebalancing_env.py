"""Tests for hourly environment dynamics."""

from __future__ import annotations

import numpy as np

from citibikerl.rebalancing.data import DemandDataset
from citibikerl.rebalancing.env import RebalancingEnv, RebalancingEnvConfig


def test_rebalancing_env_applies_demand_and_transfer_actions() -> None:
    dataset = DemandDataset(
        station_ids=("A", "B"),
        episode_days=("2026-02-01",),
        departures=np.asarray([[[0.0, 0.0], [4.0, 0.0]]]),
        arrivals=np.asarray([[[0.0, 3.0], [0.0, 0.0]]]),
    )
    env = RebalancingEnv(
        dataset,
        RebalancingEnvConfig(
            station_capacity=5,
            initial_inventory=1,
            move_amount=3,
            served_reward=1.0,
            unmet_penalty=2.0,
            move_penalty_per_bike=0.0,
            overflow_penalty=0.0,
        ),
    )

    observation = env.reset()
    assert observation.day_of_week == 6
    assert observation.is_weekend == 1
    assert observation.month_of_year == 2
    assert observation.is_holiday == 0
    assert observation.temperature_c == 0.0
    assert observation.precipitation_mm == 0.0
    assert observation.inventory == (1, 1)

    observation, reward, done, info = env.step(0)
    assert reward == 0.0
    assert not done
    assert info["moved_bikes"] == 0
    assert observation.day_of_week == 6
    assert observation.is_weekend == 1
    assert observation.month_of_year == 2
    assert observation.is_holiday == 0
    assert observation.inventory == (1, 4)

    move_action = env.actions.index((1, 0))
    observation, reward, done, info = env.step(move_action)
    assert done
    assert reward == 4.0
    assert info["served_trips"] == 4.0
    assert info["unmet_demand"] == 0.0
    assert observation.inventory == (0, 1)
