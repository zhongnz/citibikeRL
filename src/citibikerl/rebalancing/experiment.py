"""High-level experiment orchestration over the rebalancing pipeline."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import DemandDataset, split_demand_dataset_by_day, split_demand_dataset_temporal
from .env import RebalancingEnvConfig
from .io import load_model, save_model
from .policies import DemandProfilePolicy, ForecastHeuristicPolicy, NoOpPolicy, QTablePolicy
from .q_learning import (
    TrainingConfig,
    build_q_state_encoder,
    evaluate_policy,
    summarize_metrics,
    train_q_learning,
)
from .context import summarize_weather_context


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Standard output locations for a named experiment run."""

    model_path: Path
    training_metrics_path: Path
    evaluation_metrics_path: Path
    saved_policy_metrics_path: Path
    selected_stations_path: Path
    reward_plot_path: Path
    comparison_plot_path: Path
    summary_path: Path


def build_output_paths(output_prefix: str, outputs_root: str | Path = "outputs") -> ExperimentArtifacts:
    """Create standard output file paths for a named experiment."""
    root = Path(outputs_root)
    prefix = output_prefix.strip()
    if not prefix:
        raise ValueError("Output prefix must not be empty.")

    return ExperimentArtifacts(
        model_path=root / "models" / f"{prefix}_q_learning_model.json",
        training_metrics_path=root / "tables" / f"{prefix}_training_metrics.csv",
        evaluation_metrics_path=root / "tables" / f"{prefix}_policy_evaluation.csv",
        saved_policy_metrics_path=root / "tables" / f"{prefix}_saved_policy_evaluation.csv",
        selected_stations_path=root / "tables" / f"{prefix}_selected_stations.csv",
        reward_plot_path=root / "figures" / f"{prefix}_training_reward_curve.png",
        comparison_plot_path=root / "figures" / f"{prefix}_policy_comparison.png",
        summary_path=root / "logs" / f"{prefix}_experiment_summary.json",
    )


def run_experiment(
    *,
    input_path: str | Path,
    weather_input: str | Path | None,
    dataset: DemandDataset,
    station_summary: pd.DataFrame,
    env_config: RebalancingEnvConfig,
    training_config: TrainingConfig,
    output_paths: ExperimentArtifacts,
) -> dict[str, object]:
    """Run training, evaluation, serialization, plotting, and summary generation."""
    _ensure_output_dirs(output_paths)

    dataset_split = (
        split_demand_dataset_by_day(dataset, training_config.test_start_day)
        if training_config.test_start_day
        else split_demand_dataset_temporal(dataset, training_config.train_fraction)
    )
    train_dataset = dataset_split.train_dataset
    test_dataset = dataset_split.test_dataset
    training_result = train_q_learning(train_dataset, env_config, training_config)
    q_state_encoder = build_q_state_encoder(
        actions=training_result.actions,
        env_config=env_config,
        training_config=training_config,
        demand_profile=training_result.demand_profile,
        state_representation=training_result.state_representation,
    )
    forecast_fallback_policy = ForecastHeuristicPolicy()
    save_model(
        output_paths.model_path,
        station_ids=dataset.station_ids,
        q_table=training_result.q_table,
        state_visit_counts=training_result.state_visit_counts,
        actions=training_result.actions,
        env_config=env_config,
        training_config=training_config,
        state_representation=training_result.state_representation,
        demand_profile=training_result.demand_profile,
    )

    training_frame = pd.DataFrame(training_result.metrics)
    training_frame.to_csv(output_paths.training_metrics_path, index=False)
    station_summary.to_csv(output_paths.selected_stations_path, index=False)

    evaluation_frames: list[pd.DataFrame] = []
    evaluation_summaries: dict[str, dict[str, object]] = {}
    for split_name, split_dataset in _iter_named_splits(train_dataset, test_dataset):
        baseline_metrics = _tag_split(
            evaluate_policy(split_dataset, env_config, NoOpPolicy(), policy_name="baseline_no_op"),
            split_name,
        )
        heuristic_metrics = _tag_split(
            evaluate_policy(
                split_dataset,
                env_config,
                DemandProfilePolicy(
                    actions=training_result.actions,
                    demand_profile=training_result.demand_profile,
                    bucket_size=training_config.bucket_size,
                    station_capacity=env_config.station_capacity,
                    move_amount=env_config.move_amount,
                ),
                bucket_size=training_config.bucket_size,
                policy_name="heuristic_demand_profile",
            ),
            split_name,
        )
        trained_metrics = _tag_split(
            evaluate_policy(
                split_dataset,
                env_config,
                QTablePolicy(
                    training_result.q_table,
                    state_visit_counts=training_result.state_visit_counts,
                    min_visit_count=training_config.min_state_visit_count,
                    fallback_policy=forecast_fallback_policy,
                ),
                bucket_size=training_config.bucket_size,
                policy_name="q_policy_with_heuristic_fallback",
                state_encoder=q_state_encoder,
            ),
            split_name,
        )
        evaluation_frames.extend(
            [pd.DataFrame(baseline_metrics), pd.DataFrame(heuristic_metrics), pd.DataFrame(trained_metrics)],
        )
        evaluation_summaries[split_name] = {
            "episode_count": split_dataset.num_episodes,
            "episode_days": list(split_dataset.episode_days),
            "baseline_summary": summarize_metrics(baseline_metrics),
            "heuristic_summary": summarize_metrics(heuristic_metrics),
            "trained_summary": summarize_metrics(trained_metrics),
            "q_policy_with_heuristic_fallback_summary": summarize_metrics(trained_metrics),
        }

    evaluation_frame = pd.concat(evaluation_frames, ignore_index=True)
    evaluation_frame.to_csv(output_paths.evaluation_metrics_path, index=False)

    saved_model = load_model(output_paths.model_path)
    saved_q_state_encoder = build_q_state_encoder(
        actions=saved_model.actions,
        env_config=saved_model.env_config,
        training_config=saved_model.training_config,
        demand_profile=saved_model.demand_profile,
        state_representation=saved_model.state_representation,
    )
    saved_forecast_fallback_policy = ForecastHeuristicPolicy()
    saved_frames: list[pd.DataFrame] = []
    for split_name, split_dataset in _iter_named_splits(train_dataset, test_dataset):
        baseline_metrics = _tag_split(
            evaluate_policy(
                split_dataset,
                saved_model.env_config,
                NoOpPolicy(),
                bucket_size=saved_model.training_config.bucket_size,
                policy_name="baseline_no_op",
            ),
            split_name,
        )
        heuristic_metrics = _tag_split(
            evaluate_policy(
                split_dataset,
                saved_model.env_config,
                DemandProfilePolicy(
                    actions=saved_model.actions,
                    demand_profile=saved_model.demand_profile or training_result.demand_profile,
                    bucket_size=saved_model.training_config.bucket_size,
                    station_capacity=saved_model.env_config.station_capacity,
                    move_amount=saved_model.env_config.move_amount,
                ),
                bucket_size=saved_model.training_config.bucket_size,
                policy_name="heuristic_demand_profile",
            ),
            split_name,
        )
        saved_metrics = _tag_split(
            evaluate_policy(
                split_dataset,
                saved_model.env_config,
                QTablePolicy(
                    saved_model.q_table,
                    state_visit_counts=saved_model.state_visit_counts,
                    min_visit_count=saved_model.training_config.min_state_visit_count,
                    fallback_policy=saved_forecast_fallback_policy,
                ),
                bucket_size=saved_model.training_config.bucket_size,
                policy_name="saved_q_policy_with_heuristic_fallback",
                state_encoder=saved_q_state_encoder,
            ),
            split_name,
        )
        saved_frames.extend(
            [pd.DataFrame(baseline_metrics), pd.DataFrame(heuristic_metrics), pd.DataFrame(saved_metrics)],
        )
        evaluation_summaries[split_name]["saved_policy_summary"] = summarize_metrics(saved_metrics)
        evaluation_summaries[split_name]["saved_q_policy_with_heuristic_fallback_summary"] = summarize_metrics(saved_metrics)

    saved_frame = pd.concat(saved_frames, ignore_index=True)
    saved_frame.to_csv(output_paths.saved_policy_metrics_path, index=False)

    from .reporting import plot_policy_comparison, plot_training_rewards

    plot_training_rewards(output_paths.training_metrics_path, output_paths.reward_plot_path)
    plot_policy_comparison(output_paths.evaluation_metrics_path, output_paths.comparison_plot_path)

    primary_eval_split = "test" if test_dataset is not None else "train"
    primary_summary = evaluation_summaries[primary_eval_split]

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(Path(input_path)),
        "weather_input": None if weather_input is None else str(Path(weather_input)),
        "weather_context": None if weather_input is None else summarize_weather_context(weather_input),
        "selected_station_ids": list(dataset.station_ids),
        "station_count": dataset.num_stations,
        "demand_episode_count": dataset.num_episodes,
        "train_episode_count": train_dataset.num_episodes,
        "test_episode_count": 0 if test_dataset is None else test_dataset.num_episodes,
        "primary_eval_split": primary_eval_split,
        "split_strategy": "explicit_day_boundary" if training_config.test_start_day else "fractional_chronological",
        "git_head": _git_head(),
        "environment": asdict(env_config),
        "training": asdict(training_config),
        "baseline_summary": primary_summary["baseline_summary"],
        "heuristic_summary": primary_summary["heuristic_summary"],
        "trained_summary": primary_summary["trained_summary"],
        "q_policy_with_heuristic_fallback_summary": primary_summary["q_policy_with_heuristic_fallback_summary"],
        "saved_policy_summary": primary_summary["saved_policy_summary"],
        "saved_q_policy_with_heuristic_fallback_summary": primary_summary[
            "saved_q_policy_with_heuristic_fallback_summary"
        ],
        "evaluation_summaries": evaluation_summaries,
        "outputs": {
            "model": str(output_paths.model_path),
            "training_metrics": str(output_paths.training_metrics_path),
            "evaluation_metrics": str(output_paths.evaluation_metrics_path),
            "saved_policy_metrics": str(output_paths.saved_policy_metrics_path),
            "selected_stations": str(output_paths.selected_stations_path),
            "reward_plot": str(output_paths.reward_plot_path),
            "comparison_plot": str(output_paths.comparison_plot_path),
            "summary": str(output_paths.summary_path),
        },
    }
    output_paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ensure_output_dirs(output_paths: ExperimentArtifacts) -> None:
    for path in output_paths.__dict__.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _iter_named_splits(
    train_dataset: DemandDataset,
    test_dataset: DemandDataset | None,
) -> list[tuple[str, DemandDataset]]:
    splits = [("train", train_dataset)]
    if test_dataset is not None:
        splits.append(("test", test_dataset))
    return splits


def _tag_split(metrics: list[dict[str, float | int | str]], split_name: str) -> list[dict[str, float | int | str]]:
    return [{**metric, "split": split_name} for metric in metrics]


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
