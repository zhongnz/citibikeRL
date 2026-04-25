"""Tabular Q-learning over discretized station inventory states."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .data import DemandDataset
from .env import Action, Observation, RebalancingEnv, RebalancingEnvConfig
from .profile import DemandProfile, build_demand_profile

State = tuple[int, ...]
StateEncoder = Callable[[Observation], State]

INVENTORY_STATE_REPRESENTATION = "inventory_calendar_v1"
FORECAST_STATE_REPRESENTATION_V1 = "forecast_profile_v1"
FORECAST_STATE_REPRESENTATION_V2 = "forecast_profile_v2"
FORECAST_STATE_REPRESENTATION_V3 = "forecast_profile_v3"
FORECAST_STATE_REPRESENTATION = "forecast_profile_v4"


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters for tabular Q-learning."""

    train_fraction: float = 0.75
    test_start_day: str | None = None
    episodes: int = 300
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.35
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    bucket_size: int = 2
    forecast_bucket_size: int = 2
    heuristic_exploration_bias: float = 0.0
    min_state_visit_count: int = 1
    seed: int = 7


@dataclass(frozen=True)
class TrainingResult:
    """Learned Q-table, training metrics, and the associated action space."""

    q_table: dict[State, np.ndarray]
    state_visit_counts: dict[State, int]
    metrics: list[dict[str, float | int | str]]
    actions: tuple[tuple[int, int] | None, ...]
    demand_profile: DemandProfile
    state_representation: str


class Policy(Protocol):
    """Common policy interface for evaluation."""

    def select_action(self, state: State, action_count: int) -> int:
        """Select an action index for the encoded state."""


def train_q_learning(
    dataset: DemandDataset,
    env_config: RebalancingEnvConfig,
    training_config: TrainingConfig,
) -> TrainingResult:
    """Train a tabular Q-learning policy on daily demand episodes."""
    env = RebalancingEnv(dataset, env_config)
    demand_profile = build_demand_profile(dataset)
    inventory_state_encoder = build_inventory_state_encoder(
        bucket_size=training_config.bucket_size,
        station_capacity=env_config.station_capacity,
    )
    state_encoder = build_q_state_encoder(
        actions=env.actions,
        env_config=env_config,
        training_config=training_config,
        demand_profile=demand_profile,
        state_representation=FORECAST_STATE_REPRESENTATION,
    )
    heuristic_policy = build_demand_profile_policy(
        actions=env.actions,
        env_config=env_config,
        demand_profile=demand_profile,
        bucket_size=training_config.bucket_size,
    )
    rng = np.random.default_rng(training_config.seed)
    q_table: defaultdict[State, np.ndarray] = defaultdict(lambda: np.zeros(env.num_actions, dtype=float))
    state_visit_counts: defaultdict[State, int] = defaultdict(int)
    epsilon = training_config.epsilon
    metrics: list[dict[str, float | int | str]] = []

    for training_episode in range(training_config.episodes):
        demand_episode = int(rng.integers(dataset.num_episodes))
        observation = env.reset(demand_episode)
        state = state_encoder(observation)

        total_reward = 0.0
        served_trips = 0.0
        unmet_demand = 0.0
        moved_bikes = 0
        overflow_bikes = 0.0
        exploratory_actions = 0
        guided_exploration_actions = 0
        heuristic_match_actions = 0

        done = False
        while not done:
            state_visit_counts[state] += 1
            heuristic_action = heuristic_policy.select_action(
                inventory_state_encoder(observation),
                env.num_actions,
            )
            action, was_exploration, used_guidance = _select_epsilon_greedy_action(
                q_table,
                state,
                env.num_actions,
                epsilon,
                rng,
                guided_action=heuristic_action,
                guided_action_probability=training_config.heuristic_exploration_bias,
            )
            next_observation, reward, done, info = env.step(action)
            next_state = state_encoder(next_observation)
            best_next_value = 0.0 if done else float(np.max(q_table[next_state]))
            td_target = reward + training_config.gamma * best_next_value
            q_table[state][action] += training_config.alpha * (td_target - q_table[state][action])

            total_reward += reward
            served_trips += float(info["served_trips"])
            unmet_demand += float(info["unmet_demand"])
            moved_bikes += int(info["moved_bikes"])
            overflow_bikes += float(info["overflow_bikes"])
            exploratory_actions += int(was_exploration)
            guided_exploration_actions += int(used_guidance)
            heuristic_match_actions += int(action == heuristic_action)
            observation = next_observation
            state = next_state

        metrics.append(
            {
                "training_episode": training_episode,
                "demand_episode": demand_episode,
                "day": dataset.episode_days[demand_episode],
                "epsilon": epsilon,
                "total_reward": total_reward,
                "served_trips": served_trips,
                "unmet_demand": unmet_demand,
                "moved_bikes": moved_bikes,
                "overflow_bikes": overflow_bikes,
                "exploratory_actions": exploratory_actions,
                "guided_exploration_actions": guided_exploration_actions,
                "heuristic_match_actions": heuristic_match_actions,
            },
        )
        epsilon = max(training_config.epsilon_min, epsilon * training_config.epsilon_decay)

    return TrainingResult(
        q_table=dict(q_table),
        state_visit_counts=dict(state_visit_counts),
        metrics=metrics,
        actions=env.actions,
        demand_profile=demand_profile,
        state_representation=FORECAST_STATE_REPRESENTATION,
    )


def evaluate_policy(
    dataset: DemandDataset,
    env_config: RebalancingEnvConfig,
    policy: Policy,
    *,
    bucket_size: int = 2,
    policy_name: str = "policy",
    state_encoder: StateEncoder | None = None,
) -> list[dict[str, float | int | str]]:
    """Roll out a policy over every available demand episode."""
    env = RebalancingEnv(dataset, env_config)
    encoder = state_encoder or build_inventory_state_encoder(
        bucket_size=bucket_size,
        station_capacity=env_config.station_capacity,
    )
    metrics: list[dict[str, float | int | str]] = []

    for demand_episode in range(dataset.num_episodes):
        observation = env.reset(demand_episode)
        state = encoder(observation)
        fallback_actions_before = _counter_value(policy, "fallback_count")
        trusted_q_actions_before = _counter_value(policy, "trusted_q_count")
        total_reward = 0.0
        served_trips = 0.0
        unmet_demand = 0.0
        moved_bikes = 0
        overflow_bikes = 0.0
        action_count = 0

        done = False
        while not done:
            action = policy.select_action(state, env.num_actions)
            next_observation, reward, done, info = env.step(action)
            state = encoder(next_observation)
            action_count += 1
            total_reward += reward
            served_trips += float(info["served_trips"])
            unmet_demand += float(info["unmet_demand"])
            moved_bikes += int(info["moved_bikes"])
            overflow_bikes += float(info["overflow_bikes"])

        metrics.append(
            {
                "policy": policy_name,
                "demand_episode": demand_episode,
                "day": dataset.episode_days[demand_episode],
                "total_reward": total_reward,
                "served_trips": served_trips,
                "unmet_demand": unmet_demand,
                "moved_bikes": moved_bikes,
                "overflow_bikes": overflow_bikes,
                "action_count": action_count,
                "fallback_actions": _counter_value(policy, "fallback_count") - fallback_actions_before,
                "trusted_q_actions": _counter_value(policy, "trusted_q_count") - trusted_q_actions_before,
            },
        )

    return metrics


def summarize_metrics(metrics: list[dict[str, float | int | str]]) -> dict[str, float]:
    """Compute average evaluation metrics for quick console summaries."""
    if not metrics:
        return {
            "avg_reward": 0.0,
            "avg_served_trips": 0.0,
            "avg_unmet_demand": 0.0,
            "avg_moved_bikes": 0.0,
            "avg_overflow_bikes": 0.0,
            "avg_fallback_actions": 0.0,
            "avg_trusted_q_actions": 0.0,
        }

    return {
        "avg_reward": float(np.mean([float(metric["total_reward"]) for metric in metrics])),
        "avg_served_trips": float(np.mean([float(metric["served_trips"]) for metric in metrics])),
        "avg_unmet_demand": float(np.mean([float(metric["unmet_demand"]) for metric in metrics])),
        "avg_moved_bikes": float(np.mean([float(metric["moved_bikes"]) for metric in metrics])),
        "avg_overflow_bikes": float(np.mean([float(metric["overflow_bikes"]) for metric in metrics])),
        "avg_fallback_actions": float(np.mean([float(metric.get("fallback_actions", 0.0)) for metric in metrics])),
        "avg_trusted_q_actions": float(np.mean([float(metric.get("trusted_q_actions", 0.0)) for metric in metrics])),
    }


def _counter_value(policy: Policy, counter_name: str) -> int:
    return int(getattr(policy, counter_name, 0))


def encode_state(
    time_index: int,
    day_of_week: int,
    is_weekend: int,
    inventory: tuple[int, ...],
    *,
    bucket_size: int,
    station_capacity: int,
) -> State:
    """Discretize inventory values and append the current time index."""
    bucketed_inventory = tuple(
        int(min(max(value, 0), station_capacity) // bucket_size) for value in inventory
    )
    return (int(time_index), int(day_of_week), int(is_weekend), *bucketed_inventory)


def encode_forecast_state(
    time_index: int,
    day_of_week: int,
    is_weekend: int,
    month_of_year: int,
    is_holiday: int,
    temperature_c: float,
    precipitation_mm: float,
    snowfall_mm: float,
    wind_speed_m_s: float,
    inventory: tuple[int, ...],
    *,
    actions: tuple[Action, ...],
    demand_profile: DemandProfile,
    bucket_size: int,
    forecast_bucket_size: int,
    station_capacity: int,
    move_amount: int,
    heuristic_uses_bucketed_inventory: bool = True,
) -> State:
    """Encode a compact demand-aware state for Q-learning."""
    base_state = _encode_forecast_state_v3(
        time_index,
        day_of_week,
        is_weekend,
        month_of_year,
        inventory,
        actions=actions,
        demand_profile=demand_profile,
        bucket_size=bucket_size,
        forecast_bucket_size=forecast_bucket_size,
        station_capacity=station_capacity,
        move_amount=move_amount,
        heuristic_uses_bucketed_inventory=heuristic_uses_bucketed_inventory,
    )
    return (
        *base_state[:5],
        int(is_holiday),
        _bucket_temperature(temperature_c),
        _bucket_precipitation(precipitation_mm),
        _bucket_snowfall(snowfall_mm),
        _bucket_wind_speed(wind_speed_m_s),
        *base_state[5:],
    )


def _encode_forecast_state_v3(
    time_index: int,
    day_of_week: int,
    is_weekend: int,
    month_of_year: int,
    inventory: tuple[int, ...],
    *,
    actions: tuple[Action, ...],
    demand_profile: DemandProfile,
    bucket_size: int,
    forecast_bucket_size: int,
    station_capacity: int,
    move_amount: int,
    heuristic_uses_bucketed_inventory: bool = True,
) -> State:
    """Month-aware forecast state used by legacy v3 models."""
    inventory_array = np.clip(np.asarray(inventory, dtype=float), 0.0, float(station_capacity))
    expected_departures = demand_profile.departures[day_of_week, time_index]
    expected_arrivals = demand_profile.arrivals[day_of_week, time_index]
    expected_balance = inventory_array + expected_arrivals - expected_departures

    source = int(np.argmax(expected_balance))
    destination = int(np.argmin(expected_balance))
    source_inventory_bucket = _bucket_value(
        inventory_array[source],
        bucket_size=bucket_size,
        max_value=station_capacity,
    )
    destination_inventory_bucket = _bucket_value(
        inventory_array[destination],
        bucket_size=bucket_size,
        max_value=station_capacity,
    )
    source_surplus = max(float(expected_balance[source]), 0.0)
    destination_shortage = max(float(-expected_balance[destination]), 0.0)
    source_surplus_bucket = _bucket_value(
        source_surplus,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity,
    )
    destination_shortage_bucket = _bucket_value(
        destination_shortage,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity,
    )
    route_pressure_bucket = _bucket_value(
        source_surplus + destination_shortage,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity * 2,
    )
    heuristic_inventory = (
        np.minimum(np.floor(inventory_array / bucket_size) * bucket_size, station_capacity)
        if heuristic_uses_bucketed_inventory
        else inventory_array
    )
    heuristic_expected_balance = heuristic_inventory + expected_arrivals - expected_departures
    heuristic_source = int(np.argmax(heuristic_expected_balance))
    heuristic_destination = int(np.argmin(heuristic_expected_balance))
    heuristic_source_surplus = max(float(heuristic_expected_balance[heuristic_source]), 0.0)
    heuristic_destination_shortage = max(float(-heuristic_expected_balance[heuristic_destination]), 0.0)
    heuristic_action_index = _suggest_profile_action_index(
        actions=actions,
        inventory=heuristic_inventory,
        station_capacity=station_capacity,
        move_amount=move_amount,
        source=heuristic_source,
        destination=heuristic_destination,
        source_surplus=heuristic_source_surplus,
        destination_shortage=heuristic_destination_shortage,
    )
    return (
        int(time_index),
        int(day_of_week),
        int(is_weekend),
        heuristic_action_index,
        int(month_of_year),
        source,
        destination,
        source_inventory_bucket,
        destination_inventory_bucket,
        source_surplus_bucket,
        destination_shortage_bucket,
        route_pressure_bucket,
    )


def _encode_forecast_state_v1(
    time_index: int,
    day_of_week: int,
    is_weekend: int,
    inventory: tuple[int, ...],
    *,
    actions: tuple[Action, ...],
    demand_profile: DemandProfile,
    bucket_size: int,
    forecast_bucket_size: int,
    station_capacity: int,
    move_amount: int,
) -> State:
    """Legacy forecast-aware state kept for backwards-compatible model loading."""
    inventory_array = np.clip(np.asarray(inventory, dtype=float), 0.0, float(station_capacity))
    expected_departures = demand_profile.departures[day_of_week, time_index]
    expected_arrivals = demand_profile.arrivals[day_of_week, time_index]
    expected_balance = inventory_array + expected_arrivals - expected_departures

    source = int(np.argmax(expected_balance))
    destination = int(np.argmin(expected_balance))
    source_inventory_bucket = _bucket_value(
        inventory_array[source],
        bucket_size=bucket_size,
        max_value=station_capacity,
    )
    destination_inventory_bucket = _bucket_value(
        inventory_array[destination],
        bucket_size=bucket_size,
        max_value=station_capacity,
    )
    source_surplus = max(float(expected_balance[source]), 0.0)
    destination_shortage = max(float(-expected_balance[destination]), 0.0)
    source_surplus_bucket = _bucket_value(
        source_surplus,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity,
    )
    destination_shortage_bucket = _bucket_value(
        destination_shortage,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity,
    )
    route_pressure_bucket = _bucket_value(
        source_surplus + destination_shortage,
        bucket_size=forecast_bucket_size,
        max_value=station_capacity * 2,
    )
    heuristic_action_index = _suggest_profile_action_index(
        actions=actions,
        inventory=inventory_array,
        station_capacity=station_capacity,
        move_amount=move_amount,
        source=source,
        destination=destination,
        source_surplus=source_surplus,
        destination_shortage=destination_shortage,
    )
    return (
        int(time_index),
        int(day_of_week),
        int(is_weekend),
        heuristic_action_index,
        source,
        destination,
        source_inventory_bucket,
        destination_inventory_bucket,
        source_surplus_bucket,
        destination_shortage_bucket,
        route_pressure_bucket,
    )


def build_inventory_state_encoder(*, bucket_size: int, station_capacity: int) -> StateEncoder:
    """Create the baseline inventory/calendar encoder used by non-learned policies."""

    def encoder(observation: Observation) -> State:
        return encode_state(
            observation.time_index,
            observation.day_of_week,
            observation.is_weekend,
            observation.inventory,
            bucket_size=bucket_size,
            station_capacity=station_capacity,
        )

    return encoder


def build_forecast_state_encoder(
    *,
    actions: tuple[Action, ...],
    demand_profile: DemandProfile,
    bucket_size: int,
    forecast_bucket_size: int,
    station_capacity: int,
    move_amount: int,
    include_month_of_year: bool = True,
    include_exogenous_features: bool = True,
    heuristic_uses_bucketed_inventory: bool = True,
) -> StateEncoder:
    """Create the forecast-aware encoder used by the Q-table."""

    def encoder(observation: Observation) -> State:
        if include_month_of_year:
            if include_exogenous_features:
                return encode_forecast_state(
                    observation.time_index,
                    observation.day_of_week,
                    observation.is_weekend,
                    observation.month_of_year,
                    observation.is_holiday,
                    observation.temperature_c,
                    observation.precipitation_mm,
                    observation.snowfall_mm,
                    observation.wind_speed_m_s,
                    observation.inventory,
                    actions=actions,
                    demand_profile=demand_profile,
                    bucket_size=bucket_size,
                    forecast_bucket_size=forecast_bucket_size,
                    station_capacity=station_capacity,
                    move_amount=move_amount,
                    heuristic_uses_bucketed_inventory=heuristic_uses_bucketed_inventory,
                )
            return _encode_forecast_state_v3(
                observation.time_index,
                observation.day_of_week,
                observation.is_weekend,
                observation.month_of_year,
                observation.inventory,
                actions=actions,
                demand_profile=demand_profile,
                bucket_size=bucket_size,
                forecast_bucket_size=forecast_bucket_size,
                station_capacity=station_capacity,
                move_amount=move_amount,
                heuristic_uses_bucketed_inventory=heuristic_uses_bucketed_inventory,
            )
        return _encode_forecast_state_v1(
            observation.time_index,
            observation.day_of_week,
            observation.is_weekend,
            observation.inventory,
            actions=actions,
            demand_profile=demand_profile,
            bucket_size=bucket_size,
            forecast_bucket_size=forecast_bucket_size,
            station_capacity=station_capacity,
            move_amount=move_amount,
        )

    return encoder


def build_q_state_encoder(
    *,
    actions: tuple[Action, ...],
    env_config: RebalancingEnvConfig,
    training_config: TrainingConfig,
    demand_profile: DemandProfile | None,
    state_representation: str,
) -> StateEncoder:
    """Reconstruct the correct state encoder for a trained Q-table."""
    if state_representation == INVENTORY_STATE_REPRESENTATION:
        return build_inventory_state_encoder(
            bucket_size=training_config.bucket_size,
            station_capacity=env_config.station_capacity,
        )
    if state_representation == FORECAST_STATE_REPRESENTATION_V1:
        if demand_profile is None:
            raise ValueError("Forecast-aware state encoding requires a saved demand profile.")
        return build_forecast_state_encoder(
            actions=actions,
            demand_profile=demand_profile,
            bucket_size=training_config.bucket_size,
            forecast_bucket_size=training_config.forecast_bucket_size,
            station_capacity=env_config.station_capacity,
            move_amount=env_config.move_amount,
            include_month_of_year=False,
            heuristic_uses_bucketed_inventory=False,
        )
    if state_representation == FORECAST_STATE_REPRESENTATION_V2:
        if demand_profile is None:
            raise ValueError("Forecast-aware state encoding requires a saved demand profile.")
        return build_forecast_state_encoder(
            actions=actions,
            demand_profile=demand_profile,
            bucket_size=training_config.bucket_size,
            forecast_bucket_size=training_config.forecast_bucket_size,
            station_capacity=env_config.station_capacity,
            move_amount=env_config.move_amount,
            include_month_of_year=True,
            include_exogenous_features=False,
            heuristic_uses_bucketed_inventory=False,
        )
    if state_representation == FORECAST_STATE_REPRESENTATION_V3:
        if demand_profile is None:
            raise ValueError("Forecast-aware state encoding requires a saved demand profile.")
        return build_forecast_state_encoder(
            actions=actions,
            demand_profile=demand_profile,
            bucket_size=training_config.bucket_size,
            forecast_bucket_size=training_config.forecast_bucket_size,
            station_capacity=env_config.station_capacity,
            move_amount=env_config.move_amount,
            include_month_of_year=True,
            include_exogenous_features=False,
        )
    if state_representation == FORECAST_STATE_REPRESENTATION:
        if demand_profile is None:
            raise ValueError("Forecast-aware state encoding requires a saved demand profile.")
        return build_forecast_state_encoder(
            actions=actions,
            demand_profile=demand_profile,
            bucket_size=training_config.bucket_size,
            forecast_bucket_size=training_config.forecast_bucket_size,
            station_capacity=env_config.station_capacity,
            move_amount=env_config.move_amount,
            include_month_of_year=True,
        )
    raise ValueError(f"Unknown state representation: {state_representation}")


def build_demand_profile_policy(
    *,
    actions: tuple[Action, ...],
    env_config: RebalancingEnvConfig,
    demand_profile: DemandProfile,
    bucket_size: int,
) -> Policy:
    """Construct the demand-profile policy used for baselines and guided exploration."""
    from .policies import DemandProfilePolicy

    return DemandProfilePolicy(
        actions=actions,
        demand_profile=demand_profile,
        bucket_size=bucket_size,
        station_capacity=env_config.station_capacity,
        move_amount=env_config.move_amount,
    )


def _select_epsilon_greedy_action(
    q_table: defaultdict[State, np.ndarray],
    state: State,
    action_count: int,
    epsilon: float,
    rng: np.random.Generator,
    *,
    guided_action: int | None = None,
    guided_action_probability: float = 0.0,
) -> tuple[int, bool, bool]:
    if rng.random() < epsilon:
        if guided_action is not None and rng.random() < guided_action_probability:
            return int(guided_action), True, True
        return int(rng.integers(action_count)), True, False
    return int(np.argmax(q_table[state])), False, False


def _bucket_value(value: float, *, bucket_size: int, max_value: int) -> int:
    clipped = min(max(value, 0.0), float(max_value))
    return int(clipped // bucket_size)


def _bucket_temperature(temperature_c: float) -> int:
    clipped = min(max(float(temperature_c), -15.0), 35.0)
    return int((clipped + 15.0) // 5.0)


def _bucket_precipitation(precipitation_mm: float) -> int:
    value = max(float(precipitation_mm), 0.0)
    if value == 0.0:
        return 0
    if value <= 2.5:
        return 1
    if value <= 10.0:
        return 2
    return 3


def _bucket_snowfall(snowfall_mm: float) -> int:
    value = max(float(snowfall_mm), 0.0)
    if value == 0.0:
        return 0
    if value <= 5.0:
        return 1
    if value <= 25.0:
        return 2
    return 3


def _bucket_wind_speed(wind_speed_m_s: float) -> int:
    clipped = min(max(float(wind_speed_m_s), 0.0), 20.0)
    return int(clipped // 2.0)


def _suggest_profile_action_index(
    *,
    actions: tuple[Action, ...],
    inventory: np.ndarray,
    station_capacity: int,
    move_amount: int,
    source: int,
    destination: int,
    source_surplus: float,
    destination_shortage: float,
) -> int:
    if source == destination:
        return 0
    if source_surplus <= 0 or destination_shortage <= 0:
        return 0
    if source_surplus < move_amount or inventory[source] < move_amount:
        return 0
    if inventory[destination] >= station_capacity - move_amount:
        return 0
    for action_index, action in enumerate(actions):
        if action == (source, destination):
            return action_index
    return 0
