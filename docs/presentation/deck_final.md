% Citi Bike Rebalancing With RL and Heuristic Baselines
% March 2026

# Problem

- Empty origin stations create unmet trips.
- Full destination stations create overflow and operational waste.
- Rebalancing is sequential:
  moving bikes now can help later demand, but every move has a cost.
- The project goal is to test whether RL can beat a strong forecast-based heuristic under a realistic temporal holdout.

# Research Question

- Can RL outperform a demand-aware heuristic on a strict future-month test set?
- Why this matters:
  beating a no-op baseline is easy to overstate, but beating a forecast-aware operational baseline is meaningful.
- Final evaluation:
  train on January 2025 through January 2026 and test on February 2026.

# Environment Setup

- Five-station Jersey City subproblem:
  `JC115`, `HB101`, `HB106`, `JC009`, `JC109`
- State:
  inventory, calendar, holiday/weather, and demand-profile features
- Actions:
  `no_op` plus fixed-size directed bike transfers
- Reward:
  served-trip reward minus unmet-demand, move, and overflow penalties

# Data

- Citi Bike Jersey City monthly trip files from January 2025 through February 2026
- NOAA daily weather from station `USW00014734`
- 14 processed monthly flow files
- 423 daily demand episodes total
- 396 train episodes and 27 February 2026 test episodes in the primary holdout

# Methods

- `baseline_no_op`
- `heuristic_demand_profile`
- Tabular Q-learning progression:
  inventory only, then calendar, forecast, holiday, and weather context
- Dueling Double DQN with dense state features
- DQN move-margin regularization:
  only move when `Q(move) >= Q(no_op) + margin`

# Development Result

- On the early within-February split, the heuristic immediately became the real benchmark.
- Calendar features helped tabular Q-learning close the gap to no-op.
- But the demand-profile heuristic still won clearly.

![](outputs/figures/jc_202602_top5_calendar_heuristic_v1_policy_comparison.png){ width=90% }

# Main Seasonal Holdout Result

- Strict holdout:
  train through January 2026, test on February 2026
- No-op baseline: `109.33`
- Heuristic baseline: `122.45`
- Weather-aware tabular Q: `122.21`

![](outputs/figures/jc_2025_full_year_to_202602_holdout_weather_v1_policy_comparison.png){ width=90% }

# DQN Result and Caveat

- Unregularized DQN over-moved bikes and underperformed the heuristic.
- Move-margin regularization improved the best DQN run:
  `123.63` vs heuristic `122.45`
- But that win was not stable across seeds.

![](outputs/figures/jc_2025_full_year_to_202602_holdout_dqn_margin_v1_policy_comparison.png){ width=90% }

# Robustness Check

| Seed | DQN avg reward | Heuristic avg reward |
|---|---:|---:|
| 7 | 123.63 | 122.45 |
| 11 | 113.61 | 122.45 |
| 19 | 116.95 | 122.45 |

- Conclusion:
  the regularized DQN is promising, but not robust enough to claim a stable improvement.

# Takeaways

- The repo now contains a complete, reproducible RL research pipeline with real demand and weather data.
- The demand-profile heuristic is the strongest robust policy on the February 2026 holdout.
- The move-margin DQN best run is encouraging, but stability is still the main blocker.
- Next modeling step:
  residual learning around the heuristic, plus broader seed and holdout evaluation.
