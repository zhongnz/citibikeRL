# Backup Slides

## Backup 1 — Final Station Set

- `JC115`
- `HB101`
- `HB106`
- `JC009`
- `JC109`

Selected by total activity over the training window in `outputs/tables/jc_2025_full_year_to_202602_holdout_weather_v1_selected_stations.csv`.

## Backup 2 — Why the Heuristic Is Strong

- It directly uses weekday-hour demand structure from the training split.
- It moves bikes only when there is a clear forecast surplus and shortage.
- In this reduced MDP, that already captures most of the useful signal.

## Backup 3 — What the DQN Regularizer Does

- Compute Q-values for all actions.
- Compare the best move action against `no_op`.
- If the move does not beat `no_op` by the margin, execute `no_op` instead.

This reduces gratuitous transfers caused by small or noisy Q-value differences.
