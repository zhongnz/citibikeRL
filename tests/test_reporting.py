"""Tests for report figure generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from citibikerl.rebalancing.reporting import plot_policy_comparison, plot_training_rewards


def test_plot_generation_writes_png_files(tmp_path: Path) -> None:
    training_csv = tmp_path / "training.csv"
    evaluation_csv = tmp_path / "evaluation.csv"
    reward_plot = tmp_path / "reward.png"
    comparison_plot = tmp_path / "comparison.png"

    pd.DataFrame(
        [
            {"training_episode": 0, "total_reward": 1.0},
            {"training_episode": 1, "total_reward": 2.5},
            {"training_episode": 2, "total_reward": 3.0},
        ],
    ).to_csv(training_csv, index=False)
    pd.DataFrame(
        [
            {"policy": "baseline_no_op", "total_reward": -2.0, "served_trips": 1.0, "unmet_demand": 2.0, "moved_bikes": 0.0},
            {"policy": "saved_q_policy", "total_reward": 3.0, "served_trips": 4.0, "unmet_demand": 0.0, "moved_bikes": 2.0},
        ],
    ).to_csv(evaluation_csv, index=False)

    plot_training_rewards(training_csv, reward_plot)
    plot_policy_comparison(evaluation_csv, comparison_plot)

    assert reward_plot.exists()
    assert comparison_plot.exists()
    assert reward_plot.stat().st_size > 0
    assert comparison_plot.stat().st_size > 0
