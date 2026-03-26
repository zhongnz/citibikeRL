"""A small-scale bike rebalancing simulator over hourly demand episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DemandDataset
from .context import DailyContext, build_daily_context

Action = tuple[int, int] | None


@dataclass(frozen=True)
class RebalancingEnvConfig:
    """Environment parameters shared by training and evaluation."""

    station_capacity: int = 20
    initial_inventory: int = 10
    move_amount: int = 3
    served_reward: float = 1.0
    unmet_penalty: float = 2.0
    move_penalty_per_bike: float = 0.05
    overflow_penalty: float = 0.5


@dataclass(frozen=True)
class Observation:
    """Observable environment state before the next action is taken."""

    time_index: int
    day_of_week: int
    is_weekend: int
    month_of_year: int
    is_holiday: int
    temperature_c: float
    precipitation_mm: float
    snowfall_mm: float
    wind_speed_m_s: float
    inventory: tuple[int, ...]


class RebalancingEnv:
    """Simulate hourly bike demand with optional inter-station bike transfers."""

    def __init__(self, dataset: DemandDataset, config: RebalancingEnvConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.day_features: tuple[DailyContext, ...] = dataset.daily_context or build_daily_context(dataset.episode_days)
        self.actions: tuple[Action, ...] = (None,) + tuple(
            (source, destination)
            for source in range(dataset.num_stations)
            for destination in range(dataset.num_stations)
            if source != destination
        )
        self.inventory = np.full(dataset.num_stations, config.initial_inventory, dtype=float)
        self.episode_index = 0
        self.time_index = 0

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def reset(self, episode_index: int = 0) -> Observation:
        """Reset to the start of a selected demand episode."""
        if not 0 <= episode_index < self.dataset.num_episodes:
            raise IndexError(f"Episode index out of range: {episode_index}")

        self.episode_index = episode_index
        self.time_index = 0
        self.inventory = np.full(self.dataset.num_stations, self.config.initial_inventory, dtype=float)
        return self._observation()

    def step(self, action_index: int) -> tuple[Observation, float, bool, dict[str, float | int | str]]:
        """Apply one rebalancing action followed by one hour of demand."""
        if not 0 <= action_index < self.num_actions:
            raise IndexError(f"Action index out of range: {action_index}")
        if self.time_index >= self.dataset.horizon:
            raise RuntimeError("Episode already completed. Call reset() before stepping again.")

        moved_bikes = self._apply_action(self.actions[action_index])

        departures = self.dataset.departures[self.episode_index, self.time_index]
        arrivals = self.dataset.arrivals[self.episode_index, self.time_index]
        served = np.minimum(self.inventory, departures)
        unmet = departures - served

        self.inventory = self.inventory - served + arrivals
        overflow = np.maximum(self.inventory - self.config.station_capacity, 0.0)
        self.inventory = np.minimum(self.inventory, self.config.station_capacity)

        reward = (
            float(served.sum()) * self.config.served_reward
            - float(unmet.sum()) * self.config.unmet_penalty
            - float(moved_bikes) * self.config.move_penalty_per_bike
            - float(overflow.sum()) * self.config.overflow_penalty
        )

        info = {
            "served_trips": float(served.sum()),
            "unmet_demand": float(unmet.sum()),
            "moved_bikes": int(moved_bikes),
            "overflow_bikes": float(overflow.sum()),
            "episode_index": self.episode_index,
            "day": self.dataset.episode_days[self.episode_index],
            "time_index": self.time_index,
        }

        self.time_index += 1
        done = self.time_index >= self.dataset.horizon
        return self._observation(), reward, done, info

    def action_label(self, action_index: int) -> str:
        """Render an action using station IDs instead of raw indices."""
        action = self.actions[action_index]
        if action is None:
            return "no_op"
        source, destination = action
        return f"{self.dataset.station_ids[source]}->{self.dataset.station_ids[destination]}"

    def _apply_action(self, action: Action) -> int:
        if action is None:
            return 0

        source, destination = action
        movable_bikes = min(
            self.config.move_amount,
            int(self.inventory[source]),
            self.config.station_capacity - int(self.inventory[destination]),
        )
        moved_bikes = max(int(movable_bikes), 0)
        if moved_bikes:
            self.inventory[source] -= moved_bikes
            self.inventory[destination] += moved_bikes
        return moved_bikes

    def _observation(self) -> Observation:
        observed_time = min(self.time_index, self.dataset.horizon - 1)
        inventory = tuple(int(value) for value in np.rint(self.inventory).tolist())
        day_context = self.day_features[self.episode_index]
        return Observation(
            time_index=observed_time,
            day_of_week=day_context.day_of_week,
            is_weekend=day_context.is_weekend,
            month_of_year=day_context.month_of_year,
            is_holiday=day_context.is_holiday,
            temperature_c=day_context.temperature_c,
            precipitation_mm=day_context.precipitation_mm,
            snowfall_mm=day_context.snowfall_mm,
            wind_speed_m_s=day_context.wind_speed_m_s,
            inventory=inventory,
        )
