"""Plotting helpers for training and evaluation artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(tempfile.gettempdir()) / "citibikerl-matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_training_rewards(training_metrics_path: str | Path, output_path: str | Path) -> None:
    """Plot raw and smoothed training rewards over Q-learning episodes."""
    frame = pd.read_csv(training_metrics_path)
    if frame.empty:
        raise ValueError("Training metrics CSV is empty.")
    if "training_episode" not in frame.columns or "total_reward" not in frame.columns:
        raise ValueError("Training metrics CSV must contain training_episode and total_reward columns.")

    rolling_window = max(1, min(20, len(frame)))
    smoothed_reward = frame["total_reward"].rolling(window=rolling_window, min_periods=1).mean()

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(frame["training_episode"], frame["total_reward"], color="#9bb7d4", alpha=0.35, label="reward")
    axis.plot(frame["training_episode"], smoothed_reward, color="#0b3954", linewidth=2.0, label="rolling mean")
    axis.set_title("Training Reward by Episode")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Reward")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_policy_comparison(evaluation_metrics_path: str | Path, output_path: str | Path) -> None:
    """Plot average reward and unmet demand by policy."""
    frame = pd.read_csv(evaluation_metrics_path)
    if frame.empty:
        raise ValueError("Evaluation metrics CSV is empty.")
    required_columns = {"policy", "total_reward", "unmet_demand"}
    if not required_columns.issubset(frame.columns):
        missing = ", ".join(sorted(required_columns - set(frame.columns)))
        raise ValueError(f"Evaluation metrics CSV is missing columns: {missing}")

    group_columns = ["policy"]
    if "split" in frame.columns:
        group_columns = ["split", "policy"]

    summary = frame.groupby(group_columns, as_index=False)[["total_reward", "served_trips", "unmet_demand", "moved_bikes"]].mean()
    if "split" in summary.columns:
        split_order = {"train": 0, "test": 1}
        summary["sort_key"] = summary["split"].map(split_order).fillna(99)
        summary["label"] = summary["split"] + "\n" + summary["policy"]
        summary = summary.sort_values(["sort_key", "policy"], kind="stable")
    else:
        summary["label"] = summary["policy"]
        summary = summary.sort_values("total_reward", ascending=False, kind="stable")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    reward_axis, unmet_axis = axes

    reward_axis.bar(summary["label"], summary["total_reward"], color="#0b3954")
    reward_axis.set_title("Average Reward")
    reward_axis.set_ylabel("Reward")
    reward_axis.tick_params(axis="x", rotation=15)
    reward_axis.grid(axis="y", alpha=0.25, linestyle="--")

    unmet_axis.bar(summary["label"], summary["unmet_demand"], color="#bf4e30")
    unmet_axis.set_title("Average Unmet Demand")
    unmet_axis.set_ylabel("Trips")
    unmet_axis.tick_params(axis="x", rotation=15)
    unmet_axis.grid(axis="y", alpha=0.25, linestyle="--")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
