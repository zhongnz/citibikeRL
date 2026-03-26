"""Load processed hourly station flows into fixed-size learning episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .context import DailyContext, build_daily_context

PROCESSED_FLOW_COLUMNS = ["hour", "start_station_id", "end_station_id", "trip_count"]
InputPathLike = str | Path
InputPathSpec = InputPathLike | Sequence[InputPathLike]


@dataclass(frozen=True)
class DemandDataset:
    """Daily demand tensors for a selected station subset."""

    station_ids: tuple[str, ...]
    episode_days: tuple[str, ...]
    departures: np.ndarray
    arrivals: np.ndarray
    daily_context: tuple[DailyContext, ...] | None = None

    def __post_init__(self) -> None:
        if self.departures.shape != self.arrivals.shape:
            raise ValueError("Departure and arrival tensors must have identical shapes.")
        if self.departures.ndim != 3:
            raise ValueError("Demand tensors must have shape (episodes, horizon, stations).")
        if len(self.station_ids) != self.departures.shape[2]:
            raise ValueError("Station IDs must match the station axis length.")
        if len(self.episode_days) != self.departures.shape[0]:
            raise ValueError("Episode labels must match the episode axis length.")
        if self.daily_context is not None and len(self.daily_context) != self.departures.shape[0]:
            raise ValueError("Daily context rows must match the episode axis length.")

    @property
    def num_episodes(self) -> int:
        return int(self.departures.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.departures.shape[1])

    @property
    def num_stations(self) -> int:
        return int(self.departures.shape[2])


@dataclass(frozen=True)
class DemandDatasetSplit:
    """Chronological split of daily demand episodes."""

    train_dataset: DemandDataset
    test_dataset: DemandDataset | None


def normalize_station_ids(raw_station_ids: str | Sequence[str] | None) -> list[str] | None:
    """Normalize station IDs from CLI or config values."""
    if raw_station_ids is None:
        return None
    if isinstance(raw_station_ids, str):
        station_ids = [part.strip() for part in raw_station_ids.split(",") if part.strip()]
    else:
        station_ids = [str(part).strip() for part in raw_station_ids if str(part).strip()]
    return station_ids or None


def load_demand_dataset(
    input_path: InputPathSpec,
    *,
    station_ids: Sequence[str] | None = None,
    top_n_stations: int = 5,
    weather_input: str | Path | None = None,
) -> DemandDataset:
    """Load processed hourly flows and expand them into daily demand episodes."""
    frame = _load_processed_flow_frame(input_path)
    selected_station_ids = _resolve_station_ids(frame, station_ids=station_ids, top_n_stations=top_n_stations)
    day_labels = sorted(frame["day"].unique())
    departures = np.zeros((len(day_labels), 24, len(selected_station_ids)), dtype=float)
    arrivals = np.zeros_like(departures)

    station_to_index = {station_id: index for index, station_id in enumerate(selected_station_ids)}
    day_to_index = {day: index for index, day in enumerate(day_labels)}

    departure_counts = (
        frame[frame["start_station_id"].isin(selected_station_ids)]
        .groupby(["day", "hour_index", "start_station_id"], sort=True)["trip_count"]
        .sum()
    )
    for (day, hour_index, station_id), trip_count in departure_counts.items():
        departures[day_to_index[str(day)], int(hour_index), station_to_index[str(station_id)]] = float(trip_count)

    arrival_counts = (
        frame[frame["end_station_id"].isin(selected_station_ids)]
        .groupby(["day", "hour_index", "end_station_id"], sort=True)["trip_count"]
        .sum()
    )
    for (day, hour_index, station_id), trip_count in arrival_counts.items():
        arrivals[day_to_index[str(day)], int(hour_index), station_to_index[str(station_id)]] = float(trip_count)

    return DemandDataset(
        station_ids=tuple(selected_station_ids),
        episode_days=tuple(str(day) for day in day_labels),
        departures=departures,
        arrivals=arrivals,
        daily_context=build_daily_context(day_labels, weather_input=weather_input),
    )


def build_station_activity_summary(
    input_path: InputPathSpec,
    *,
    station_ids: Sequence[str] | None = None,
    top_n_stations: int = 5,
) -> pd.DataFrame:
    """Summarize total departures, arrivals, and activity for selected stations."""
    frame = _load_processed_flow_frame(input_path)
    selected_station_ids = _resolve_station_ids(frame, station_ids=station_ids, top_n_stations=top_n_stations)
    departures = frame.groupby("start_station_id")["trip_count"].sum()
    arrivals = frame.groupby("end_station_id")["trip_count"].sum()

    summary = pd.DataFrame(
        {
            "station_id": selected_station_ids,
            "total_departures": [float(departures.get(station_id, 0.0)) for station_id in selected_station_ids],
            "total_arrivals": [float(arrivals.get(station_id, 0.0)) for station_id in selected_station_ids],
        },
    )
    summary["total_activity"] = summary["total_departures"] + summary["total_arrivals"]
    summary["selection_rank"] = range(1, len(summary) + 1)
    return summary


def select_demand_episodes(dataset: DemandDataset, episode_indices: Sequence[int]) -> DemandDataset:
    """Select a subset of daily episodes from a demand dataset."""
    if not episode_indices:
        raise ValueError("Episode selection must contain at least one episode.")

    episode_index_array = np.asarray(episode_indices, dtype=int)
    if np.any(episode_index_array < 0) or np.any(episode_index_array >= dataset.num_episodes):
        raise IndexError("Episode selection contains out-of-range indices.")

    return DemandDataset(
        station_ids=dataset.station_ids,
        episode_days=tuple(dataset.episode_days[index] for index in episode_index_array.tolist()),
        departures=dataset.departures[episode_index_array].copy(),
        arrivals=dataset.arrivals[episode_index_array].copy(),
        daily_context=None
        if dataset.daily_context is None
        else tuple(dataset.daily_context[index] for index in episode_index_array.tolist()),
    )


def split_demand_dataset_temporal(dataset: DemandDataset, train_fraction: float) -> DemandDatasetSplit:
    """Split a demand dataset chronologically into train and held-out test episodes."""
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in the interval (0, 1].")
    if dataset.num_episodes < 2 or train_fraction == 1.0:
        return DemandDatasetSplit(train_dataset=dataset, test_dataset=None)

    train_count = int(np.floor(dataset.num_episodes * train_fraction))
    train_count = min(max(train_count, 1), dataset.num_episodes - 1)

    train_dataset = select_demand_episodes(dataset, list(range(train_count)))
    test_dataset = select_demand_episodes(dataset, list(range(train_count, dataset.num_episodes)))
    return DemandDatasetSplit(train_dataset=train_dataset, test_dataset=test_dataset)


def split_demand_dataset_by_day(dataset: DemandDataset, test_start_day: str) -> DemandDatasetSplit:
    """Split a demand dataset by an explicit day boundary."""
    split_day = str(test_start_day).strip()
    if not split_day:
        raise ValueError("test_start_day must not be empty.")

    train_indices = [index for index, day in enumerate(dataset.episode_days) if day < split_day]
    test_indices = [index for index, day in enumerate(dataset.episode_days) if day >= split_day]
    if not train_indices:
        raise ValueError(f"No training episodes fall before test_start_day={split_day}.")
    if not test_indices:
        raise ValueError(f"No test episodes fall on or after test_start_day={split_day}.")

    return DemandDatasetSplit(
        train_dataset=select_demand_episodes(dataset, train_indices),
        test_dataset=select_demand_episodes(dataset, test_indices),
    )


def _resolve_station_ids(
    frame: pd.DataFrame,
    *,
    station_ids: Sequence[str] | None,
    top_n_stations: int,
) -> list[str]:
    if station_ids:
        selected_station_ids = [str(station_id) for station_id in station_ids]
    else:
        departures = frame.groupby("start_station_id")["trip_count"].sum()
        arrivals = frame.groupby("end_station_id")["trip_count"].sum()
        activity = departures.add(arrivals, fill_value=0.0).sort_values(ascending=False, kind="stable")
        selected_station_ids = [str(station_id) for station_id in activity.index if str(station_id)][:top_n_stations]

    if not selected_station_ids:
        raise ValueError("No stations were selected from the processed flow file.")

    return selected_station_ids


def normalize_input_paths(raw_input_paths: InputPathSpec) -> list[Path]:
    """Normalize one or many processed CSV inputs."""
    if isinstance(raw_input_paths, (str, Path)):
        parts = [part.strip() for part in str(raw_input_paths).split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw_input_paths if str(part).strip()]

    if not parts:
        raise ValueError("At least one input path is required.")
    return [Path(part) for part in parts]


def _load_processed_flow_frame(input_path: InputPathSpec) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for data_path in normalize_input_paths(input_path):
        if not data_path.exists():
            raise FileNotFoundError(f"Processed flow file not found: {data_path}")

        frame = pd.read_csv(
            data_path,
            dtype={
                "start_station_id": "string",
                "end_station_id": "string",
            },
        )
        missing_columns = [column for column in PROCESSED_FLOW_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Processed flow file is missing columns: {', '.join(missing_columns)}")
        if frame.empty:
            raise ValueError(f"Processed flow file is empty: {data_path}")
        frame["hour"] = pd.to_datetime(frame["hour"], errors="raise")
        expected_year_month = _extract_expected_year_month(data_path)
        if expected_year_month is not None:
            frame = frame[frame["hour"].dt.strftime("%Y-%m") == expected_year_month].copy()
            if frame.empty:
                raise ValueError(
                    f"Processed flow file has no rows matching its month {expected_year_month}: {data_path}",
                )
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    frame["trip_count"] = pd.to_numeric(frame["trip_count"], errors="raise")
    frame["start_station_id"] = frame["start_station_id"].fillna("").astype(str)
    frame["end_station_id"] = frame["end_station_id"].fillna("").astype(str)
    frame["day"] = frame["hour"].dt.strftime("%Y-%m-%d")
    frame["hour_index"] = frame["hour"].dt.hour.astype(int)
    return frame.sort_values(["hour", "start_station_id", "end_station_id"], kind="stable").reset_index(drop=True)


def _extract_expected_year_month(data_path: Path) -> str | None:
    match = re.search(r"jc_(\d{4})(\d{2})_hourly_flows\.csv$", data_path.name, flags=re.IGNORECASE)
    if match is None:
        return None
    year, month = match.groups()
    return f"{year}-{month}"
