"""Simple policies used for baseline and learned evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import Action
from .profile import DemandProfile
from .q_learning import Policy, State


@dataclass(frozen=True)
class NoOpPolicy:
    """Always select the no-op action."""

    def select_action(self, state: State, action_count: int) -> int:
        del state, action_count
        return 0


@dataclass(frozen=True)
class ForecastHeuristicPolicy:
    """Fallback policy that reads the heuristic action embedded in forecast-aware states."""

    heuristic_action_position: int = 3

    def select_action(self, state: State, action_count: int) -> int:
        if len(state) <= self.heuristic_action_position:
            return 0
        action = int(state[self.heuristic_action_position])
        if not 0 <= action < action_count:
            return 0
        return action


@dataclass(frozen=True)
class QTablePolicy:
    """Greedy policy derived from a learned Q-table."""

    q_table: dict[State, np.ndarray]
    state_visit_counts: dict[State, int] | None = None
    min_visit_count: int = 1
    fallback_policy: Policy | None = None

    def select_action(self, state: State, action_count: int) -> int:
        values = self.q_table.get(state)
        visit_count = self.min_visit_count if self.state_visit_counts is None else int(self.state_visit_counts.get(state, 0))
        if values is None or visit_count < self.min_visit_count:
            if self.fallback_policy is not None:
                return int(self.fallback_policy.select_action(state, action_count))
            return 0
        if len(values) != action_count:
            raise ValueError("Q-table action count does not match the environment.")
        return int(np.argmax(values))


@dataclass(frozen=True)
class DemandProfilePolicy:
    """Move bikes from expected surplus stations toward expected shortage stations."""

    actions: tuple[Action, ...]
    demand_profile: DemandProfile
    bucket_size: int
    station_capacity: int
    move_amount: int

    def __post_init__(self) -> None:
        action_to_index = {}
        for index, action in enumerate(self.actions):
            action_to_index[action] = index
        object.__setattr__(self, "_action_to_index", action_to_index)

    def select_action(self, state: State, action_count: int) -> int:
        if len(self.actions) != action_count:
            raise ValueError("Action count does not match the policy action map.")

        time_index, day_of_week, is_weekend, *inventory_buckets = state
        del is_weekend

        inventory = np.minimum(np.asarray(inventory_buckets, dtype=float) * self.bucket_size, self.station_capacity)
        expected_departures = self.demand_profile.departures[day_of_week, time_index]
        expected_arrivals = self.demand_profile.arrivals[day_of_week, time_index]
        expected_balance = inventory + expected_arrivals - expected_departures

        source = int(np.argmax(expected_balance))
        destination = int(np.argmin(expected_balance))
        if source == destination:
            return 0

        source_surplus = float(expected_balance[source])
        destination_shortage = float(-expected_balance[destination])
        if source_surplus <= 0 or destination_shortage <= 0:
            return 0
        if source_surplus < self.move_amount or inventory[source] < self.move_amount:
            return 0
        if inventory[destination] >= self.station_capacity - self.move_amount:
            return 0

        return int(self._action_to_index.get((source, destination), 0))
