# Citi Bike Rebalancing With Real Demand Data: Draft v1

## Abstract

This project studies a small bike-rebalancing problem built from real Jersey City Citi Bike trips. The repo implements a reproducible pipeline for hourly demand preprocessing, environment simulation, baseline evaluation, tabular Q-learning, and a dueling Double DQN with weather and holiday context. On a strict holdout that trains on January 1, 2025 through January 31, 2026 and tests on February 1, 2026 through February 28, 2026, the strongest robust result is still a simple demand-profile heuristic. A move-margin-regularized DQN beats that heuristic in the best run, but the improvement is not stable across seeds.

## 1. Introduction

Bike-share systems suffer when local station inventories drift away from demand. Empty origin stations create unmet trips, while full destination stations create overflow and operational waste. That makes station rebalancing a sequential decision problem: moving bikes now can improve later service, but every move also costs time and capacity.

This repository studies a deliberately compact version of that problem. Instead of attempting system-wide dispatch, it focuses on five high-activity Jersey City stations and a fixed transfer action size. The goal is not to ship a production optimizer. The goal is to test whether reinforcement learning can outperform simple, demand-aware policies once the evaluation is made temporally realistic.

## 2. Problem Setup

The environment models hourly bike demand over a five-station subset. The final station set used in the primary seasonal holdout is:

- `JC115`
- `HB101`
- `HB106`
- `JC009`
- `JC109`

These stations were selected by total start-plus-end activity over the training window in [jc_2025_full_year_to_202602_holdout_weather_v1_selected_stations.csv](../../outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_selected_stations.csv).

At each step, the policy chooses either `no_op` or a directed transfer of a fixed number of bikes from one station to another. The reward function combines:

- positive reward for served trips,
- penalty for unmet demand,
- penalty per bike moved,
- penalty for overflow.

This formulation intentionally rewards service improvement while making unnecessary movement expensive.

## 3. Data and Preprocessing

Trip demand comes from official Jersey City Citi Bike monthly files covering January 2025 through February 2026, documented in [CITIBIKE_DATA_SOURCE.md](../../references/datasets/CITIBIKE_DATA_SOURCE.md). Raw trips are validated and aggregated into hourly origin-destination flows under `data/processed/jc_YYYYMM_hourly_flows.csv`.

The primary experiment uses all 14 processed monthly files. The resulting dataset contains 423 daily demand episodes, of which 396 are used for training and 27 February 2026 episodes are used for testing. Daily weather context comes from NOAA station `USW00014734`, documented in [NOAA_WEATHER_SOURCE.md](../../references/datasets/NOAA_WEATHER_SOURCE.md).

Exploratory analysis is captured in [01_data_overview.ipynb](../../notebooks/01_data_overview.ipynb).

## 4. Methods

Three policy families are compared.

`baseline_no_op`

This baseline never moves bikes. It is intentionally weak but gives a lower bound on operational value.

`heuristic_demand_profile`

This policy estimates average departures and arrivals by weekday and hour from the training split. At decision time, it computes expected next-hour surpluses and shortages, then transfers bikes from the largest surplus station to the largest shortage station when the move is feasible. This is a one-step, forecast-aware heuristic.

RL methods

The tabular path starts with simple inventory states, then adds calendar features, forecast-profile features, and finally holiday/weather context. The deep RL path replaces the Q-table with a dense dueling Double DQN over calendar, inventory, weather, and demand-profile features.

The main regularization added to the DQN is a `no_op` margin gate implemented in [dqn.py](../../src/citibikerl/rebalancing/dqn.py). The policy only takes a move action when the predicted Q-value for that move exceeds the `no_op` Q-value by a configured margin. Otherwise it chooses `no_op`. This targets the main failure mode seen in early DQN runs: over-moving bikes on weak value differences.

## 5. Evaluation Protocol

The final evaluation is chronological.

- Train window: January 1, 2025 through January 31, 2026
- Test window: February 1, 2026 through February 28, 2026
- Train episodes: 396
- Test episodes: 27

This split is recorded in [jc_2025_full_year_to_202602_holdout_weather_v1_experiment_summary.json](../../outputs/logs/jc_2025_full_year_to_202602_holdout_weather_v1_experiment_summary.json).

Primary metrics are:

- average reward,
- average served trips,
- average unmet demand,
- average bikes moved,
- average overflow bikes.

## 6. Results

### 6.1 Development progression

The early within-February split shows why the heuristic became the key benchmark. On the 21-train-day / 7-test-day setup, the initial tabular Q-policy underperformed `baseline_no_op` (`121.12` vs `122.50` average reward). Adding calendar features brought the tabular policy up to `122.48`, nearly matching the no-op baseline but still far below the heuristic. Once the demand-profile heuristic was added, it reached `127.47` average reward on the same split. These results are recorded in:

- [jc_202602_top5_split_v1_policy_evaluation.csv](../../outputs/tables/jc_202602_top5_split_v1_policy_evaluation.csv)
- [jc_202602_top5_calendar_v1_policy_evaluation.csv](../../outputs/tables/jc_202602_top5_calendar_v1_policy_evaluation.csv)
- [jc_202602_top5_calendar_heuristic_v1_policy_evaluation.csv](../../outputs/tables/jc_202602_top5_calendar_heuristic_v1_policy_evaluation.csv)

### 6.2 Seasonal holdout

The more important result is the year-to-February holdout. On that split:

| Policy | Avg reward | Avg unmet demand | Avg bikes moved |
|---|---:|---:|---:|
| No-op baseline | 109.33 | 12.48 | 0.00 |
| Demand-profile heuristic | 122.45 | 8.07 | 9.11 |
| Q table with heuristic fallback | 122.21 | 8.15 | 9.22 |

These numbers come from [jc_2025_full_year_to_202602_holdout_weather_v1_policy_evaluation.csv](../../outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_policy_evaluation.csv). The tabular Q policy nearly matches the heuristic because the strict future-month states are almost all unseen: it averages `23.96` fallback actions and only `0.04` trusted Q-table actions per 24-hour holdout episode.

### 6.3 DQN and move regularization

The unregularized DQN improves representation power but introduces a new problem: it moves too many bikes.

| Policy | Avg reward | Avg unmet demand | Avg bikes moved |
|---|---:|---:|---:|
| Demand-profile heuristic | 122.45 | 8.07 | 9.11 |
| DQN without move margin | 122.12 | 8.19 | 44.63 |
| DQN with move margin, seed 7 | 123.63 | 8.30 | 20.07 |

The unregularized DQN result is in [jc_2025_full_year_to_202602_holdout_dqn_v2_policy_evaluation.csv](../../outputs/tables/jc_2025_full_year_to_202602_holdout_dqn_v2_policy_evaluation.csv). The regularized result is in [jc_2025_full_year_to_202602_holdout_dqn_margin_v1_policy_evaluation.csv](../../outputs/tables/jc_2025_full_year_to_202602_holdout_dqn_margin_v1_policy_evaluation.csv).

The important detail is that the margin gate improves reward by cutting back gratuitous transfers. It does not improve served trips; in fact, the seed-7 regularized DQN serves slightly fewer trips than the heuristic. It wins because it trades off movement and overflow more effectively in that run.

### 6.4 Robustness check

A three-seed sweep shows that the regularized DQN result is not stable. On the same February 2026 holdout, the trained DQN rewards are:

- seed 7: `123.63`
- seed 11: `113.61`
- seed 19: `116.95`

The baseline and heuristic are unchanged across those runs, and the heuristic remains `122.45`. The sweep artifact is [jc_2025_full_year_to_202602_holdout_dqn_margin_seed_sweep.csv](../../outputs/tables/jc_2025_full_year_to_202602_holdout_dqn_margin_seed_sweep.csv).

This changes the conclusion. The seed-7 run is a promising best case, not a robust policy win.

## 7. Discussion

The demand-profile heuristic is difficult to beat because it already captures the highest-signal structure in this reduced MDP: hourly directional imbalance conditioned on calendar context. The tabular learner eventually approaches that heuristic once calendar, forecast, holiday, and weather features are added, but it pays a sample-efficiency penalty.

The DQN improves generalization but exposes a different problem. Without extra control, it overestimates small advantages and moves bikes too aggressively. The move-margin gate is effective because it turns low-confidence move decisions into `no_op`. That addresses the visible failure mode, but the seed sweep shows that optimization is still unstable.

The strongest research conclusion is therefore not "DQN solved the problem." It is that a simple forecast-aware heuristic is a powerful benchmark, and any RL method must be judged against it under chronological holdout and repeated seeds.

## 8. Limitations and Future Work

This project still simplifies real operations heavily.

- Only five stations are modeled.
- Actions are pairwise fixed-size transfers, not truck routes.
- Weather comes from one station and there are no event-level demand features.
- DQN robustness was checked on only three seeds.

The most useful next steps are:

1. Train a residual learner around the heuristic rather than a full replacement policy.
2. Expand the seed sweep and, if available, extend the holdout to more future months.
3. Add richer exogenous features such as holidays, events, and route-level travel costs.

## 9. Conclusion

The repository now supports a complete, reproducible rebalancing research workflow with real Citi Bike demand, weather context, chronological evaluation, and report-ready artifacts. The current evidence says that the demand-profile heuristic is the strongest reliable method on the seasonal holdout. A regularized DQN can outperform it in the best run, but that gain is not yet robust enough to claim a stable improvement.
