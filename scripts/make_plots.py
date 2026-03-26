#!/usr/bin/env python3
"""Generate report-ready figures from training and evaluation metrics."""

from __future__ import annotations

import argparse

from citibikerl.rebalancing import plot_policy_comparison, plot_training_rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures from training and evaluation metrics.")
    parser.add_argument("--training-metrics", required=True, help="Training metrics CSV from train_q_learning.py")
    parser.add_argument("--evaluation-metrics", required=True, help="Evaluation metrics CSV")
    parser.add_argument("--reward-plot", required=True, help="Output PNG path for the training reward curve")
    parser.add_argument("--comparison-plot", required=True, help="Output PNG path for the policy comparison chart")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_training_rewards(args.training_metrics, args.reward_plot)
    plot_policy_comparison(args.evaluation_metrics, args.comparison_plot)
    print(f"Wrote reward plot to: {args.reward_plot}")
    print(f"Wrote comparison plot to: {args.comparison_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
