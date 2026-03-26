"""Demand-profile utilities derived from training episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .data import DemandDataset


@dataclass(frozen=True)
class DemandProfile:
    """Average departures and arrivals by weekday and hour."""

    departures: np.ndarray
    arrivals: np.ndarray


def build_demand_profile(dataset: DemandDataset) -> DemandProfile:
    """Estimate expected hourly departures and arrivals from observed training episodes."""
    num_stations = dataset.num_stations
    horizon = dataset.horizon
    departure_sums = np.zeros((7, horizon, num_stations), dtype=float)
    arrival_sums = np.zeros_like(departure_sums)
    weekday_counts = np.zeros(7, dtype=int)

    for episode_index, day_label in enumerate(dataset.episode_days):
        weekday = date.fromisoformat(day_label).weekday()
        departure_sums[weekday] += dataset.departures[episode_index]
        arrival_sums[weekday] += dataset.arrivals[episode_index]
        weekday_counts[weekday] += 1

    overall_departures = dataset.departures.mean(axis=0)
    overall_arrivals = dataset.arrivals.mean(axis=0)
    departure_profile = np.zeros_like(departure_sums)
    arrival_profile = np.zeros_like(arrival_sums)

    for weekday in range(7):
        if weekday_counts[weekday] > 0:
            departure_profile[weekday] = departure_sums[weekday] / weekday_counts[weekday]
            arrival_profile[weekday] = arrival_sums[weekday] / weekday_counts[weekday]
        else:
            departure_profile[weekday] = overall_departures
            arrival_profile[weekday] = overall_arrivals

    return DemandProfile(departures=departure_profile, arrivals=arrival_profile)
