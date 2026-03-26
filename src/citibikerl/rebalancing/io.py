"""Serialize and load learned Q-tables for later evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .env import Action, RebalancingEnvConfig
from .profile import DemandProfile
from .q_learning import State, TrainingConfig


@dataclass(frozen=True)
class SavedModel:
    """Serialized tabular Q-learning policy and metadata."""

    station_ids: tuple[str, ...]
    actions: tuple[Action, ...]
    q_table: dict[State, np.ndarray]
    state_visit_counts: dict[State, int]
    env_config: RebalancingEnvConfig
    training_config: TrainingConfig
    state_representation: str
    demand_profile: DemandProfile | None


def save_model(
    output_path: str | Path,
    *,
    station_ids: tuple[str, ...],
    q_table: dict[State, np.ndarray],
    state_visit_counts: dict[State, int],
    actions: tuple[Action, ...],
    env_config: RebalancingEnvConfig,
    training_config: TrainingConfig,
    state_representation: str,
    demand_profile: DemandProfile | None = None,
) -> None:
    """Write a Q-table and its metadata to JSON."""
    payload = {
        "station_ids": list(station_ids),
        "actions": [None if action is None else list(action) for action in actions],
        "q_table": {state_to_key(state): values.tolist() for state, values in q_table.items()},
        "state_visit_counts": {state_to_key(state): int(count) for state, count in state_visit_counts.items()},
        "environment": asdict(env_config),
        "training": asdict(training_config),
        "state_representation": state_representation,
        "demand_profile": None
        if demand_profile is None
        else {
            "departures": demand_profile.departures.tolist(),
            "arrivals": demand_profile.arrivals.tolist(),
        },
    }

    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(input_path: str | Path) -> SavedModel:
    """Load a serialized Q-table from JSON."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    q_table = {
        key_to_state(state_key): np.asarray(values, dtype=float)
        for state_key, values in payload["q_table"].items()
    }
    state_visit_counts = {
        key_to_state(state_key): int(count)
        for state_key, count in payload.get("state_visit_counts", {}).items()
    }
    actions = tuple(None if action is None else (int(action[0]), int(action[1])) for action in payload["actions"])
    demand_profile_payload = payload.get("demand_profile")
    demand_profile = None
    if demand_profile_payload is not None:
        demand_profile = DemandProfile(
            departures=np.asarray(demand_profile_payload["departures"], dtype=float),
            arrivals=np.asarray(demand_profile_payload["arrivals"], dtype=float),
        )
    return SavedModel(
        station_ids=tuple(payload["station_ids"]),
        actions=actions,
        q_table=q_table,
        state_visit_counts=state_visit_counts,
        env_config=RebalancingEnvConfig(**payload["environment"]),
        training_config=TrainingConfig(**payload["training"]),
        state_representation=payload.get("state_representation", "inventory_calendar_v1"),
        demand_profile=demand_profile,
    )


def state_to_key(state: State) -> str:
    """Serialize a tuple state into a JSON key."""
    return "|".join(str(value) for value in state)


def key_to_state(key: str) -> State:
    """Deserialize a JSON key back into a tuple state."""
    return tuple(int(value) for value in key.split("|"))
