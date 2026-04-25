# Report Outline

## Abstract
- Objective: compare simple bike-rebalancing policies on a five-station Jersey City Citi Bike subproblem.
- Methods: no-op baseline, demand-profile heuristic, tabular Q-learning variants, and a dueling Double DQN with move regularization.
- Main outcome: the heuristic is the strongest robust policy; a regularized DQN can beat it in the best run but is unstable across seeds.

## 1. Introduction
- Motivate station-level rebalancing as a sequential decision problem with asymmetric costs.
- Frame the project as a compact RL test bed rather than a production dispatch system.
- State the research question:
  can RL outperform a forecast-based heuristic on a strict month holdout?

## 2. Problem Setup
- Operational scope:
  five highest-activity Jersey City stations from the full-year training window: `JC115`, `HB101`, `HB106`, `JC009`, `JC109`.
- Environment:
  hourly demand episodes, fixed station capacity, fixed move amount, deterministic inventory transitions given demand.
- State:
  inventory, calendar context, holiday/weather context, and demand-profile features.
- Action space:
  `no_op` plus directed transfers between station pairs.
- Reward:
  served-trip reward minus unmet-demand, move, and overflow penalties.

## 3. Data and Preprocessing
- Citi Bike monthly Jersey City trip files from January 2025 through February 2026.
- Hourly aggregation into OD flow counts in `data/processed/jc_YYYYMM_hourly_flows.csv`.
- NOAA daily weather context from station `USW00014734`.
- Episode construction:
  423 daily demand episodes total, with the primary holdout using 396 train days and 27 February 2026 test days.
- EDA artifact:
  `notebooks/01_data_overview.ipynb`.

## 4. Methods
- Baseline 1:
  `baseline_no_op`.
- Baseline 2:
  `heuristic_demand_profile`, which shifts bikes from forecast surplus to forecast shortage using weekday-hour average demand.
- Tabular RL:
  chronological split, then calendar-, forecast-, and weather-aware state refinements.
- Deep RL:
  dueling Double DQN with dense state features.
- DQN regularization:
  only take a move when `Q(move) >= Q(no_op) + margin`; otherwise fall back to `no_op`.

## 5. Evaluation Protocol and Metrics
- Primary evaluation:
  train on January 2025 through January 2026, test on February 2026 (`test_start_day = 2026-02-01`).
- Secondary development evaluation:
  within-February 2026 chronological split.
- Metrics:
  average reward, served trips, unmet demand, bikes moved, overflow bikes.
- Robustness:
  seed sweep for the regularized DQN.

## 6. Results
- Early February split:
  calendar features help tabular Q-learning, but the heuristic remains much stronger.
- Full-year seasonal holdout:
  the Q table with heuristic fallback approaches the heuristic, but the fallback counters show the strict holdout is mostly unseen by the Q-table.
- DQN:
  unregularized DQN over-moves bikes; move-margin regularization improves the best run.
- Robustness:
  the regularized DQN win is not stable across seeds, so the heuristic remains the strongest robust result.

## 7. Discussion
- Explain why the heuristic is hard to beat:
  it directly encodes the most useful structure in this small MDP.
- Explain why RL struggles:
  sample efficiency, state sparsity, and sensitivity to over-moving.
- Interpret the move-margin gate as a practical control:
  it reduces gratuitous transfers but does not solve optimization instability.

## 8. Limitations and Future Work
- Only five stations and a simplified transfer action model.
- No vehicle routing or staffing constraints.
- One weather source and no event-level demand features.
- Next method steps:
  residual learning around the heuristic, more seeds/months, or a more stable function approximator.

## 9. Conclusion
- The project demonstrates a reproducible RL pipeline with real data and seasonal evaluation.
- The main substantive finding is not "deep RL wins"; it is that the demand-profile heuristic is the most reliable policy, while regularized DQN is promising but unstable.
