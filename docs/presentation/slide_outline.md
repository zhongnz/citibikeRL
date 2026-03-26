# Slide Outline

## Slide 1 — Title
- Citi Bike rebalancing with RL and heuristic baselines
- Scope:
  Jersey City five-station subproblem, January 2025 to February 2026

## Slide 2 — Why this problem matters
- Empty stations create unmet trips
- Full stations create overflow and operational cost
- Moving bikes helps only when the timing and direction are right

## Slide 3 — Research question
- Can RL outperform a demand-aware heuristic under a strict temporal holdout?
- Why a strong heuristic matters more than beating a no-op baseline

## Slide 4 — Environment setup
- State:
  inventory, calendar, holiday/weather, demand-profile features
- Action:
  `no_op` plus directed fixed-size transfers
- Reward:
  served reward minus unmet, move, and overflow penalties

## Slide 5 — Data
- Citi Bike Jersey City monthly trip files, January 2025 through February 2026
- NOAA daily weather from `USW00014734`
- Final station set:
  `JC115`, `HB101`, `HB106`, `JC009`, `JC109`

## Slide 6 — Baselines and methods
- No-op baseline
- Demand-profile heuristic
- Tabular Q-learning progression
- Dueling Double DQN with move-margin regularization

## Slide 7 — Main result: seasonal holdout
- Show `F2`
- Explain:
  heuristic is the strongest robust policy on the February 2026 holdout

## Slide 8 — DQN result and caveat
- Show `F3`
- Show seed-sweep table from `outputs/tables/jc_2025_full_year_to_202602_holdout_dqn_margin_seed_sweep.csv`
- Key message:
  regularization helps, but the DQN win is not stable across seeds

## Slide 9 — Takeaways
- Real-data pipeline is complete and reproducible
- Strong heuristic benchmark is essential
- Best DQN run is promising, but robustness is still the blocking issue

## Slide 10 — Q&A
- Backup discussion:
  why the heuristic is so strong, and what the next modeling step should be

---

## Speaker timing target
- 10 slides / 10-12 minutes total.
