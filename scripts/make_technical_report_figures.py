#!/usr/bin/env python3
"""Generate visual figures embedded by docs/report/technical_report.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("outputs/figures/technical_report")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy": "#0b3954",
    "rust": "#bf4e30",
    "amber": "#f9a825",
    "violet": "#7e57c2",
    "teal": "#26a69a",
    "gray": "#6e6e6e",
}


def fig_selected_stations() -> None:
    df = pd.read_csv(
        "outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_selected_stations.csv"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["total_departures"], 0.36, label="Departures", color=PALETTE["navy"])
    ax.bar(x + 0.18, df["total_arrivals"], 0.36, label="Arrivals", color=PALETTE["rust"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["station_id"])
    ax.set_ylabel("Total trips (Jan 2025 – Feb 2026)")
    ax.set_title("Five selected stations, by total activity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_stations.png", dpi=150)
    plt.close(fig)


def fig_demand_heatmap() -> None:
    selected = ["JC115", "HB101", "HB106", "JC009", "JC109"]
    flow_files = [Path(f"data/processed/jc_2025{m:02d}_hourly_flows.csv") for m in range(1, 13)]
    flow_files += [Path("data/processed/jc_202601_hourly_flows.csv"),
                   Path("data/processed/jc_202602_hourly_flows.csv")]
    frames = [pd.read_csv(p) for p in flow_files if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["start_station_id"].isin(selected)].copy()
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce", utc=False)
    df = df.dropna(subset=["hour"])
    df["dow"] = df["hour"].dt.dayofweek
    df["h"] = df["hour"].dt.hour
    pivot = df.groupby(["dow", "h"])["trip_count"].sum() / df.groupby("dow")["hour"].nunique().repeat(24).reset_index(drop=True).values[:len(df.groupby(["dow","h"]).size())]
    pivot = df.groupby(["dow", "h"])["trip_count"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Mean departures per (dow × hour), summed across 5 stations")
    fig.colorbar(im, ax=ax, label="Trips / hour")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "demand_heatmap.png", dpi=150)
    plt.close(fig)


def fig_reward_decomposition() -> None:
    df = pd.read_csv(
        "outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_policy_evaluation.csv"
    )
    test = df[df["split"] == "test"]
    means = test.groupby("policy")[
        ["served_trips", "unmet_demand", "moved_bikes", "overflow_bikes"]
    ].mean()

    served_r = 1.0
    unmet_p = 2.0
    move_p = 0.05
    overflow_p = 0.5

    means["served_c"] = means["served_trips"] * served_r
    means["unmet_c"] = -means["unmet_demand"] * unmet_p
    means["move_c"] = -means["moved_bikes"] * move_p
    means["overflow_c"] = -means["overflow_bikes"] * overflow_p
    means["total"] = means[["served_c", "unmet_c", "move_c", "overflow_c"]].sum(axis=1)

    order = ["baseline_no_op", "heuristic_demand_profile", "q_policy_with_heuristic_fallback"]
    labels = ["No-op", "Heuristic", "Q+fallback (v4)"]
    components = [
        ("served_c", "+1.0 · served", PALETTE["navy"]),
        ("unmet_c", "−2.0 · unmet", PALETTE["rust"]),
        ("move_c", "−0.05 · moved", PALETTE["amber"]),
        ("overflow_c", "−0.5 · overflow", PALETTE["violet"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom_pos = np.zeros(len(order))
    bottom_neg = np.zeros(len(order))
    for comp, label, color in components:
        values = np.array([means.loc[p, comp] for p in order])
        bot = np.where(values >= 0, bottom_pos, bottom_neg + values)
        height = np.abs(values)
        ax.bar(labels, height, bottom=bot, label=label, color=color)
        for i, v in enumerate(values):
            if v >= 0:
                bottom_pos[i] += v
            else:
                bottom_neg[i] += v

    for i, p in enumerate(order):
        total = means.loc[p, "total"]
        ax.text(i, bottom_pos[i] + 4, f"net = {total:.2f}",
                ha="center", fontweight="bold", color="black")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Reward contribution per episode (mean over 27 test days)")
    ax.set_title("Reward decomposition by policy (Feb-2026 holdout)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_decomposition.png", dpi=150)
    plt.close(fig)


def fig_coverage() -> None:
    v4 = json.loads(
        Path("outputs/logs/jc_2025_full_year_to_202602_holdout_weather_v1_experiment_summary.json").read_text()
    )
    v1 = json.loads(
        Path("outputs/logs/jc_2025_full_year_to_202602_holdout_coarse_v1_experiment_summary.json").read_text()
    )

    encs = ["v4 (17-tuple)", "v1 (11-tuple)"]
    trusts = [v4["trained_summary"]["avg_trusted_q_actions"], v1["trained_summary"]["avg_trusted_q_actions"]]
    fallbacks = [v4["trained_summary"]["avg_fallback_actions"], v1["trained_summary"]["avg_fallback_actions"]]
    rewards = [v4["trained_summary"]["avg_reward"], v1["trained_summary"]["avg_reward"]]
    heuristic_r = v4["heuristic_summary"]["avg_reward"]
    baseline_r = v4["baseline_summary"]["avg_reward"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(encs, trusts, label="Q-table trusted", color=PALETTE["navy"])
    axes[0].bar(encs, fallbacks, bottom=trusts, label="Heuristic fallback", color=PALETTE["rust"])
    for i, t in enumerate(trusts):
        axes[0].text(i, 24.5, f"trust = {t:.2f}/24\n({100*t/24:.1f}%)", ha="center", fontsize=9)
    axes[0].set_ylabel("Decisions per 24-hour episode")
    axes[0].set_title("Q-table consultation rate")
    axes[0].set_ylim(0, 28)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(encs, rewards, color=PALETTE["navy"])
    axes[1].axhline(heuristic_r, color=PALETTE["rust"], linestyle="--", label=f"Heuristic = {heuristic_r:.2f}")
    axes[1].axhline(baseline_r, color=PALETTE["gray"], linestyle=":", label=f"No-op = {baseline_r:.2f}")
    for i, r in enumerate(rewards):
        axes[1].text(i, r + 0.6, f"{r:.2f}", ha="center")
    axes[1].set_ylabel("Avg reward per episode")
    axes[1].set_title("Test reward (Feb-2026)")
    axes[1].set_ylim(105, 128)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Coverage vs reward: shrinking the state fixes the consultation rate but not the reward",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_experiment.png", dpi=150)
    plt.close(fig)


def fig_seed_sweep() -> None:
    df = pd.read_csv("outputs/tables/dqn_seed_sweep/dqn_repcfg_8seed_summary.csv").sort_values("seed")

    rewards = df["avg_reward"].values
    seeds = df["seed"].values
    mean_r = rewards.mean()
    se = rewards.std(ddof=1) / np.sqrt(len(rewards))
    ci_lo, ci_hi = mean_r - 1.96 * se, mean_r + 1.96 * se
    heuristic_r = 122.45
    baseline_r = 109.33

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(rewards, np.ones_like(rewards), s=120, color=PALETTE["navy"], zorder=3, alpha=0.85)
    for s, r in zip(seeds, rewards):
        ax.text(r, 1.07, f"s={s}", ha="center", fontsize=8)

    ax.axvspan(ci_lo, ci_hi, alpha=0.18, color=PALETTE["navy"],
               label=f"95% CI on mean: [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axvline(mean_r, color=PALETTE["navy"], linestyle="-", linewidth=1.4,
               label=f"DQN mean = {mean_r:.2f} (std {rewards.std(ddof=1):.2f})")
    ax.axvline(heuristic_r, color=PALETTE["rust"], linestyle="--", linewidth=1.4,
               label=f"Heuristic = {heuristic_r:.2f}")
    ax.axvline(baseline_r, color=PALETTE["gray"], linestyle=":", linewidth=1.4,
               label=f"No-op baseline = {baseline_r:.2f}")

    ax.set_yticks([])
    ax.set_xlim(105, 128)
    ax.set_xlabel("Average test reward (Feb-2026 holdout)")
    ax.set_title("DQN 8-seed sweep at report config — 1/8 seeds beats heuristic; mean is below")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dqn_seed_sweep.png", dpi=150)
    plt.close(fig)


def fig_move_distribution() -> None:
    df = pd.read_csv(
        "outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_policy_evaluation.csv"
    )
    test = df[df["split"] == "test"]

    policy_specs = [
        ("baseline_no_op", "No-op", PALETTE["gray"]),
        ("heuristic_demand_profile", "Heuristic", PALETTE["rust"]),
        ("q_policy_with_heuristic_fallback", "Q+fallback (v4)", PALETTE["navy"]),
    ]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (policy, label, color) in enumerate(policy_specs, start=1):
        values = test[test["policy"] == policy]["moved_bikes"].values
        x_jitter = i + rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(x_jitter, values, color=color, alpha=0.7, s=35, edgecolor="black", linewidth=0.4)
        ax.hlines(values.mean(), i - 0.25, i + 0.25, color="black", linewidth=2,
                  label=f"{label} mean = {values.mean():.2f}" if i == 1 else f"{label} mean = {values.mean():.2f}")

    ax.set_xticks(range(1, len(policy_specs) + 1))
    ax.set_xticklabels([s[1] for s in policy_specs])
    ax.set_ylabel("Bikes moved per 24-hour episode")
    ax.set_title("Movement intensity (Feb-2026, 27 test episodes)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "move_distribution.png", dpi=150)
    plt.close(fig)


def fig_action_space() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    stations = ["JC115", "HB101", "HB106", "JC009", "JC109"]
    angles = np.linspace(0, 2 * np.pi, len(stations), endpoint=False)
    radius = 1.0
    coords = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    for (x, y), name in zip(coords, stations):
        ax.scatter(x, y, s=900, color=PALETTE["navy"], zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, y, name, color="white", ha="center", va="center", fontsize=8.5, fontweight="bold")

    for i in range(len(stations)):
        for j in range(len(stations)):
            if i == j:
                continue
            x0, y0 = coords[i]
            x1, y1 = coords[j]
            dx, dy = x1 - x0, y1 - y0
            ax.annotate("", xy=(x0 + 0.85 * dx, y0 + 0.85 * dy),
                         xytext=(x0 + 0.15 * dx, y0 + 0.15 * dy),
                         arrowprops=dict(arrowstyle="->", color=PALETTE["gray"],
                                         alpha=0.35, lw=0.8))

    ax.scatter(0, 0, s=300, color=PALETTE["amber"], zorder=4, edgecolor="black", linewidth=1)
    ax.text(0, -0.18, "no_op", ha="center", fontsize=9, fontweight="bold")
    ax.text(0, 1.45, "Action space (21 actions): no_op + 20 directed pairs",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(0, -1.45, "Each move transfers up to 3 bikes from source to destination",
            ha="center", fontsize=9, color=PALETTE["gray"])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "action_space.png", dpi=150)
    plt.close(fig)


def fig_state_encoder_growth() -> None:
    encoders = [
        ("inventory\n(baseline)", 8, ["t", "dow", "weekend", "5x inv buckets"]),
        ("forecast v1", 11, ["t", "dow", "weekend", "h-act", "src", "dst",
                             "src-inv", "dst-inv", "surplus", "shortage", "pressure"]),
        ("forecast v2", 12, ["v1 + month"]),
        ("forecast v3", 12, ["v2 (heuristic on bucketed inv)"]),
        ("forecast v4\n(default)", 17, ["v3 + holiday + 4 weather buckets"]),
        ("DQN dense", 54, ["sin/cos t, dow OH, month OH, weather,\ninv, expected dep/arr/balance, src/dst OH"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sizes = [e[1] for e in encoders]
    labels = [e[0] for e in encoders]
    colors = [PALETTE["gray"]] + [PALETTE["navy"]] * 4 + [PALETTE["rust"]]
    bars = ax.bar(labels, sizes, color=colors)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, size + 0.8, str(size),
                ha="center", fontweight="bold")
    ax.set_ylabel("Feature dimensionality")
    ax.set_title("State-encoder progression: tabular variants stay below ~17 dims; DQN uses 54-dim dense vector")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "state_encoder_growth.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Writing figures to {FIG_DIR}")
    fig_selected_stations()
    fig_demand_heatmap()
    fig_reward_decomposition()
    fig_coverage()
    fig_seed_sweep()
    fig_move_distribution()
    fig_action_space()
    fig_state_encoder_growth()
    print("Done.")


if __name__ == "__main__":
    main()
