"""DQN-style value-function approximation over dense rebalancing state features."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .data import DemandDataset
from .env import Observation, RebalancingEnv, RebalancingEnvConfig
from .profile import DemandProfile, build_demand_profile
from .q_learning import (
    Action,
    build_demand_profile_policy,
    build_inventory_state_encoder,
    summarize_metrics,
)

DenseState = np.ndarray
DenseStateEncoder = Callable[[Observation], DenseState]

DQN_STATE_REPRESENTATION = "dense_forecast_exogenous_v1"


@dataclass(frozen=True)
class DQNTrainingConfig:
    """Hyperparameters for the NumPy DQN trainer."""

    train_fraction: float = 0.75
    test_start_day: str | None = None
    episodes: int = 400
    gamma: float = 0.99
    epsilon: float = 0.35
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 10000
    replay_warmup: int = 256
    hidden_dim: int = 64
    heuristic_bucket_size: int = 2
    target_update_interval: int = 100
    train_interval: int = 1
    gradient_clip: float = 5.0
    heuristic_exploration_bias: float = 0.25
    move_action_margin: float = 0.0
    double_dqn: bool = True
    dueling: bool = True
    seed: int = 7


@dataclass(frozen=True)
class DQNTrainingResult:
    """Trained network weights, metrics, and metadata."""

    network_state: dict[str, np.ndarray]
    metrics: list[dict[str, float | int | str]]
    actions: tuple[Action, ...]
    demand_profile: DemandProfile
    state_representation: str
    feature_dim: int


@dataclass(frozen=True)
class SavedDQNModel:
    """Serialized DQN policy and metadata."""

    station_ids: tuple[str, ...]
    actions: tuple[Action, ...]
    network_state: dict[str, np.ndarray]
    env_config: RebalancingEnvConfig
    training_config: DQNTrainingConfig
    state_representation: str
    demand_profile: DemandProfile
    feature_dim: int


@dataclass(frozen=True)
class DQNPolicy:
    """Greedy policy derived from a trained dense Q-network."""

    network_state: dict[str, np.ndarray]
    state_encoder: DenseStateEncoder
    dueling: bool = True
    move_action_margin: float = 0.0

    def select_action(self, observation: Observation, action_count: int) -> int:
        q_values = _forward_network(self.network_state, self.state_encoder(observation)[None, :], dueling=self.dueling)
        if q_values.shape[1] != action_count:
            raise ValueError("Network action count does not match the environment.")
        return _select_regularized_action(q_values[0], move_action_margin=self.move_action_margin)


def train_dqn(
    dataset: DemandDataset,
    env_config: RebalancingEnvConfig,
    training_config: DQNTrainingConfig,
) -> DQNTrainingResult:
    """Train a dueling double DQN agent on daily demand episodes."""
    env = RebalancingEnv(dataset, env_config)
    demand_profile = build_demand_profile(dataset)
    dense_state_encoder = build_dense_state_encoder(
        demand_profile=demand_profile,
        station_capacity=env_config.station_capacity,
    )
    inventory_state_encoder = build_inventory_state_encoder(
        bucket_size=training_config.heuristic_bucket_size,
        station_capacity=env_config.station_capacity,
    )
    heuristic_policy = build_demand_profile_policy(
        actions=env.actions,
        env_config=env_config,
        demand_profile=demand_profile,
        bucket_size=training_config.heuristic_bucket_size,
    )

    rng = np.random.default_rng(training_config.seed)
    feature_dim = int(dense_state_encoder(env.reset(0)).shape[0])
    online_state = _initialize_network(
        input_dim=feature_dim,
        hidden_dim=training_config.hidden_dim,
        output_dim=env.num_actions,
        rng=rng,
        dueling=training_config.dueling,
    )
    target_state = _copy_network_state(online_state)
    optimizer_state = _initialize_adam_state(online_state)
    replay_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=training_config.replay_capacity)

    epsilon = training_config.epsilon
    metrics: list[dict[str, float | int | str]] = []
    total_steps = 0

    for training_episode in range(training_config.episodes):
        demand_episode = int(rng.integers(dataset.num_episodes))
        observation = env.reset(demand_episode)
        state_vector = dense_state_encoder(observation)

        total_reward = 0.0
        served_trips = 0.0
        unmet_demand = 0.0
        moved_bikes = 0
        overflow_bikes = 0.0
        exploratory_actions = 0
        guided_exploration_actions = 0
        heuristic_match_actions = 0
        updates = 0
        loss_sum = 0.0

        done = False
        while not done:
            heuristic_action = heuristic_policy.select_action(
                inventory_state_encoder(observation),
                env.num_actions,
            )
            action, was_exploration, used_guidance = _select_dqn_action(
                online_state=online_state,
                state_vector=state_vector,
                epsilon=epsilon,
                rng=rng,
                action_count=env.num_actions,
                guided_action=heuristic_action,
                guided_action_probability=training_config.heuristic_exploration_bias,
                dueling=training_config.dueling,
            )
            next_observation, reward, done, info = env.step(action)
            next_state_vector = dense_state_encoder(next_observation)
            replay_buffer.append(
                (
                    state_vector.copy(),
                    int(action),
                    float(reward),
                    next_state_vector.copy(),
                    float(done),
                ),
            )

            if len(replay_buffer) >= training_config.replay_warmup and total_steps % training_config.train_interval == 0:
                batch = _sample_replay_batch(replay_buffer, training_config.batch_size, rng)
                loss = _train_dqn_batch(
                    online_state=online_state,
                    target_state=target_state,
                    optimizer_state=optimizer_state,
                    batch=batch,
                    learning_rate=training_config.learning_rate,
                    gamma=training_config.gamma,
                    gradient_clip=training_config.gradient_clip,
                    double_dqn=training_config.double_dqn,
                    dueling=training_config.dueling,
                )
                updates += 1
                loss_sum += float(loss)

            total_steps += 1
            if total_steps % training_config.target_update_interval == 0:
                target_state = _copy_network_state(online_state)

            total_reward += reward
            served_trips += float(info["served_trips"])
            unmet_demand += float(info["unmet_demand"])
            moved_bikes += int(info["moved_bikes"])
            overflow_bikes += float(info["overflow_bikes"])
            exploratory_actions += int(was_exploration)
            guided_exploration_actions += int(used_guidance)
            heuristic_match_actions += int(action == heuristic_action)

            observation = next_observation
            state_vector = next_state_vector

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
                "gradient_updates": updates,
                "avg_loss": 0.0 if updates == 0 else float(loss_sum / updates),
            },
        )
        epsilon = max(training_config.epsilon_min, epsilon * training_config.epsilon_decay)

    if total_steps % training_config.target_update_interval != 0:
        target_state = _copy_network_state(online_state)

    return DQNTrainingResult(
        network_state=target_state,
        metrics=metrics,
        actions=env.actions,
        demand_profile=demand_profile,
        state_representation=DQN_STATE_REPRESENTATION,
        feature_dim=feature_dim,
    )


def evaluate_dqn_policy(
    dataset: DemandDataset,
    env_config: RebalancingEnvConfig,
    policy: DQNPolicy,
    *,
    policy_name: str = "dqn_policy",
) -> list[dict[str, float | int | str]]:
    """Roll out a DQN policy across all available episodes."""
    env = RebalancingEnv(dataset, env_config)
    metrics: list[dict[str, float | int | str]] = []

    for demand_episode in range(dataset.num_episodes):
        observation = env.reset(demand_episode)
        total_reward = 0.0
        served_trips = 0.0
        unmet_demand = 0.0
        moved_bikes = 0
        overflow_bikes = 0.0
        action_count = 0

        done = False
        while not done:
            action = policy.select_action(observation, env.num_actions)
            observation, reward, done, info = env.step(action)
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
                "fallback_actions": 0,
                "trusted_q_actions": 0,
            },
        )

    return metrics


def build_dense_state_encoder(*, demand_profile: DemandProfile, station_capacity: int) -> DenseStateEncoder:
    """Create a dense feature encoder for neural value approximation."""
    demand_scale = max(
        1.0,
        float(station_capacity),
        float(np.max(demand_profile.departures)),
        float(np.max(demand_profile.arrivals)),
    )
    weather_scales = {
        "temperature": 20.0,
        "precipitation": np.log1p(50.0),
        "snowfall": np.log1p(500.0),
        "wind": 15.0,
    }

    def encoder(observation: Observation) -> DenseState:
        time_angle = 2.0 * np.pi * (float(observation.time_index) / 24.0)
        time_features = np.asarray([np.sin(time_angle), np.cos(time_angle)], dtype=float)

        weekday_features = np.zeros(7, dtype=float)
        weekday_features[int(observation.day_of_week)] = 1.0
        month_features = np.zeros(12, dtype=float)
        month_features[int(observation.month_of_year) - 1] = 1.0
        binary_features = np.asarray([float(observation.is_weekend), float(observation.is_holiday)], dtype=float)
        weather_features = np.asarray(
            [
                np.clip(float(observation.temperature_c) / weather_scales["temperature"], -2.0, 2.0),
                np.clip(np.log1p(max(float(observation.precipitation_mm), 0.0)) / weather_scales["precipitation"], 0.0, 1.5),
                np.clip(np.log1p(max(float(observation.snowfall_mm), 0.0)) / weather_scales["snowfall"], 0.0, 1.5),
                np.clip(float(observation.wind_speed_m_s) / weather_scales["wind"], 0.0, 2.0),
            ],
            dtype=float,
        )

        inventory = np.asarray(observation.inventory, dtype=float)
        inventory_norm = np.clip(inventory / float(station_capacity), 0.0, 1.0)
        expected_departures = demand_profile.departures[observation.day_of_week, observation.time_index]
        expected_arrivals = demand_profile.arrivals[observation.day_of_week, observation.time_index]
        expected_departures_norm = np.clip(expected_departures / demand_scale, 0.0, 3.0)
        expected_arrivals_norm = np.clip(expected_arrivals / demand_scale, 0.0, 3.0)
        expected_balance_norm = np.clip((inventory + expected_arrivals - expected_departures) / demand_scale, -3.0, 3.0)

        source = int(np.argmax(expected_balance_norm))
        destination = int(np.argmin(expected_balance_norm))
        source_one_hot = np.zeros_like(inventory_norm)
        destination_one_hot = np.zeros_like(inventory_norm)
        source_one_hot[source] = 1.0
        destination_one_hot[destination] = 1.0
        route_pressure = np.asarray(
            [
                np.clip(max(float(expected_balance_norm[source]), 0.0), 0.0, 3.0),
                np.clip(max(float(-expected_balance_norm[destination]), 0.0), 0.0, 3.0),
            ],
            dtype=float,
        )

        return np.concatenate(
            [
                time_features,
                weekday_features,
                month_features,
                binary_features,
                weather_features,
                inventory_norm,
                expected_departures_norm,
                expected_arrivals_norm,
                expected_balance_norm,
                source_one_hot,
                destination_one_hot,
                route_pressure,
            ],
        ).astype(float)

    return encoder


def save_dqn_model(
    output_path: str | Path,
    *,
    station_ids: tuple[str, ...],
    actions: tuple[Action, ...],
    network_state: dict[str, np.ndarray],
    env_config: RebalancingEnvConfig,
    training_config: DQNTrainingConfig,
    demand_profile: DemandProfile,
    state_representation: str,
    feature_dim: int,
) -> None:
    """Serialize a DQN policy to JSON."""
    payload = {
        "station_ids": list(station_ids),
        "actions": [None if action is None else list(action) for action in actions],
        "network_state": {name: values.tolist() for name, values in network_state.items()},
        "environment": asdict(env_config),
        "training": asdict(training_config),
        "demand_profile": {
            "departures": demand_profile.departures.tolist(),
            "arrivals": demand_profile.arrivals.tolist(),
        },
        "state_representation": state_representation,
        "feature_dim": int(feature_dim),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dqn_model(input_path: str | Path) -> SavedDQNModel:
    """Load a serialized DQN policy from JSON."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    demand_profile_payload = payload["demand_profile"]
    return SavedDQNModel(
        station_ids=tuple(payload["station_ids"]),
        actions=tuple(None if action is None else (int(action[0]), int(action[1])) for action in payload["actions"]),
        network_state={name: np.asarray(values, dtype=float) for name, values in payload["network_state"].items()},
        env_config=RebalancingEnvConfig(**payload["environment"]),
        training_config=DQNTrainingConfig(**payload["training"]),
        state_representation=payload.get("state_representation", DQN_STATE_REPRESENTATION),
        demand_profile=DemandProfile(
            departures=np.asarray(demand_profile_payload["departures"], dtype=float),
            arrivals=np.asarray(demand_profile_payload["arrivals"], dtype=float),
        ),
        feature_dim=int(payload["feature_dim"]),
    )


def _select_dqn_action(
    *,
    online_state: dict[str, np.ndarray],
    state_vector: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    action_count: int,
    guided_action: int | None,
    guided_action_probability: float,
    dueling: bool,
) -> tuple[int, bool, bool]:
    if rng.random() < epsilon:
        if guided_action is not None and rng.random() < guided_action_probability:
            return int(guided_action), True, True
        return int(rng.integers(action_count)), True, False
    q_values = _forward_network(online_state, state_vector[None, :], dueling=dueling)
    return int(np.argmax(q_values[0])), False, False


def _select_regularized_action(q_values: np.ndarray, *, move_action_margin: float) -> int:
    action = int(np.argmax(q_values))
    if action != 0 and float(q_values[action]) < float(q_values[0]) + float(move_action_margin):
        return 0
    return action


def _sample_replay_batch(
    replay_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.choice(len(replay_buffer), size=min(batch_size, len(replay_buffer)), replace=False)
    batch = [replay_buffer[int(index)] for index in indices]
    states = np.stack([row[0] for row in batch], axis=0)
    actions = np.asarray([row[1] for row in batch], dtype=int)
    rewards = np.asarray([row[2] for row in batch], dtype=float)
    next_states = np.stack([row[3] for row in batch], axis=0)
    dones = np.asarray([row[4] for row in batch], dtype=float)
    return states, actions, rewards, next_states, dones


def _train_dqn_batch(
    *,
    online_state: dict[str, np.ndarray],
    target_state: dict[str, np.ndarray],
    optimizer_state: dict[str, dict[str, np.ndarray] | int],
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    learning_rate: float,
    gamma: float,
    gradient_clip: float,
    double_dqn: bool,
    dueling: bool,
) -> float:
    states, actions, rewards, next_states, dones = batch
    next_online_q = _forward_network(online_state, next_states, dueling=dueling)
    next_target_q = _forward_network(target_state, next_states, dueling=dueling)
    if double_dqn:
        next_actions = np.argmax(next_online_q, axis=1)
        next_values = next_target_q[np.arange(len(next_actions)), next_actions]
    else:
        next_values = np.max(next_target_q, axis=1)
    targets = rewards + gamma * (1.0 - dones) * next_values

    q_values, cache = _forward_network_with_cache(online_state, states, dueling=dueling)
    predicted = q_values[np.arange(len(actions)), actions]
    td_error = predicted - targets
    loss = _huber_loss(td_error)

    d_predicted = _huber_derivative(td_error) / float(len(actions))
    dq_values = np.zeros_like(q_values)
    dq_values[np.arange(len(actions)), actions] = d_predicted
    gradients = _backward_network(online_state, cache, dq_values, dueling=dueling)
    _apply_gradients_adam(
        online_state,
        gradients,
        optimizer_state,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
    )
    return float(loss)


def _initialize_network(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    rng: np.random.Generator,
    dueling: bool,
) -> dict[str, np.ndarray]:
    scale1 = np.sqrt(2.0 / max(input_dim, 1))
    scale2 = np.sqrt(2.0 / max(hidden_dim, 1))
    state = {
        "W1": rng.normal(0.0, scale1, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=float),
        "W2": rng.normal(0.0, scale2, size=(hidden_dim, hidden_dim)),
        "b2": np.zeros(hidden_dim, dtype=float),
    }
    if dueling:
        state["Wv"] = rng.normal(0.0, scale2, size=(hidden_dim, 1))
        state["bv"] = np.zeros(1, dtype=float)
        state["Wa"] = rng.normal(0.0, scale2, size=(hidden_dim, output_dim))
        state["ba"] = np.zeros(output_dim, dtype=float)
    else:
        state["W3"] = rng.normal(0.0, scale2, size=(hidden_dim, output_dim))
        state["b3"] = np.zeros(output_dim, dtype=float)
    return state


def _copy_network_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values.copy() for name, values in state.items()}


def _initialize_adam_state(network_state: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray] | int]:
    return {
        "m": {name: np.zeros_like(values) for name, values in network_state.items()},
        "v": {name: np.zeros_like(values) for name, values in network_state.items()},
        "t": 0,
    }


def _forward_network(state: dict[str, np.ndarray], inputs: np.ndarray, *, dueling: bool) -> np.ndarray:
    q_values, _ = _forward_network_with_cache(state, inputs, dueling=dueling)
    return q_values


def _forward_network_with_cache(
    state: dict[str, np.ndarray],
    inputs: np.ndarray,
    *,
    dueling: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = inputs @ state["W1"] + state["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ state["W2"] + state["b2"]
    a2 = np.maximum(z2, 0.0)

    cache = {
        "inputs": inputs,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
    }

    if dueling:
        value = a2 @ state["Wv"] + state["bv"]
        advantage = a2 @ state["Wa"] + state["ba"]
        q_values = value + advantage - advantage.mean(axis=1, keepdims=True)
        cache["value"] = value
        cache["advantage"] = advantage
    else:
        q_values = a2 @ state["W3"] + state["b3"]
    return q_values, cache


def _backward_network(
    state: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    dq_values: np.ndarray,
    *,
    dueling: bool,
) -> dict[str, np.ndarray]:
    a2 = cache["a2"]
    gradients: dict[str, np.ndarray]

    if dueling:
        d_value = np.sum(dq_values, axis=1, keepdims=True)
        d_advantage = dq_values - dq_values.mean(axis=1, keepdims=True)
        gradients = {
            "Wv": a2.T @ d_value,
            "bv": d_value.sum(axis=0),
            "Wa": a2.T @ d_advantage,
            "ba": d_advantage.sum(axis=0),
        }
        da2 = d_value @ state["Wv"].T + d_advantage @ state["Wa"].T
    else:
        gradients = {
            "W3": a2.T @ dq_values,
            "b3": dq_values.sum(axis=0),
        }
        da2 = dq_values @ state["W3"].T

    dz2 = da2 * (cache["z2"] > 0.0)
    gradients["W2"] = cache["a1"].T @ dz2
    gradients["b2"] = dz2.sum(axis=0)

    da1 = dz2 @ state["W2"].T
    dz1 = da1 * (cache["z1"] > 0.0)
    gradients["W1"] = cache["inputs"].T @ dz1
    gradients["b1"] = dz1.sum(axis=0)
    return gradients


def _apply_gradients_adam(
    network_state: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    optimizer_state: dict[str, dict[str, np.ndarray] | int],
    *,
    learning_rate: float,
    gradient_clip: float,
) -> None:
    m = optimizer_state["m"]
    v = optimizer_state["v"]
    optimizer_state["t"] = int(optimizer_state["t"]) + 1
    t = int(optimizer_state["t"])
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    total_norm = np.sqrt(sum(float(np.sum(grad * grad)) for grad in gradients.values()))
    scale = 1.0 if total_norm <= gradient_clip or total_norm == 0.0 else gradient_clip / total_norm

    for name, grad in gradients.items():
        grad = grad * scale
        m[name] = beta1 * m[name] + (1.0 - beta1) * grad
        v[name] = beta2 * v[name] + (1.0 - beta2) * (grad * grad)
        m_hat = m[name] / (1.0 - beta1**t)
        v_hat = v[name] / (1.0 - beta2**t)
        network_state[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def _huber_loss(td_error: np.ndarray) -> float:
    abs_error = np.abs(td_error)
    quadratic = np.minimum(abs_error, 1.0)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * quadratic * quadratic + linear))


def _huber_derivative(td_error: np.ndarray) -> np.ndarray:
    derivative = td_error.copy()
    derivative[np.abs(td_error) > 1.0] = np.sign(td_error[np.abs(td_error) > 1.0])
    return derivative


__all__ = [
    "DQNPolicy",
    "DQNTrainingConfig",
    "DQNTrainingResult",
    "DQN_STATE_REPRESENTATION",
    "SavedDQNModel",
    "build_dense_state_encoder",
    "evaluate_dqn_policy",
    "load_dqn_model",
    "save_dqn_model",
    "summarize_metrics",
    "train_dqn",
]
