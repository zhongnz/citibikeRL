"""Tests for model serialization and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from citibikerl.rebalancing import DemandProfile, RebalancingEnvConfig, TrainingConfig, load_model, save_model


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    q_table = {
        (0, 1, 2): np.asarray([0.0, 1.5, -0.25]),
        (1, 2, 3): np.asarray([2.0, 0.5, 0.0]),
    }

    save_model(
        model_path,
        station_ids=("A", "B"),
        q_table=q_table,
        state_visit_counts={(0, 1, 2): 4, (1, 2, 3): 2},
        actions=(None, (0, 1), (1, 0)),
        env_config=RebalancingEnvConfig(station_capacity=8, initial_inventory=4, move_amount=2),
        training_config=TrainingConfig(episodes=10, bucket_size=1, seed=9),
        state_representation="forecast_profile_v1",
        demand_profile=DemandProfile(
            departures=np.asarray([[[1.0, 0.0]]]),
            arrivals=np.asarray([[[0.0, 1.0]]]),
        ),
    )

    saved_model = load_model(model_path)

    assert saved_model.station_ids == ("A", "B")
    assert saved_model.actions == (None, (0, 1), (1, 0))
    assert saved_model.env_config.station_capacity == 8
    assert saved_model.training_config.bucket_size == 1
    assert saved_model.state_representation == "forecast_profile_v1"
    assert saved_model.demand_profile is not None
    assert saved_model.state_visit_counts[(0, 1, 2)] == 4
    assert saved_model.demand_profile.departures.tolist() == [[[1.0, 0.0]]]
    assert saved_model.q_table[(0, 1, 2)].tolist() == [0.0, 1.5, -0.25]
