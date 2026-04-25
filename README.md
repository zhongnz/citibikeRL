# CitiBikeRL — Model-free Policy Evaluation for Small-Scale Rebalancing

This repository supports a reinforcement-learning course project on Citi Bike rebalancing.
It combines:
- **organization-first delivery** (proposal/report/presentation workflows), and
- **build-ready structure** (package skeleton, dataset scripts, validation checks).
- **first runnable RL slice** (environment simulation, baseline evaluation, tabular Q-learning).

## Project overview

### Problem
Bike-share systems can become imbalanced: some stations run out of bikes while others have excess inventory.
This hurts served trips and increases unmet demand.

### Core idea
Model hourly rebalancing decisions as an MDP, train/evaluate a tabular Q-learning policy, and compare against a **Do Not Refill** baseline.

---

## Repository map

```text
citibikeRL/
├── README.md
├── .gitignore
├── pyproject.toml
├── Makefile
├── data/
│   ├── raw/                    # immutable source datasets
│   ├── processed/              # transformed/aggregated data
│   └── external/               # auxiliary data assets
├── docs/
│   ├── proposal/               # proposal drafts/final + checklist
│   ├── report/                 # report drafts/final + section plan
│   ├── presentation/           # slide outline/deck/speaker notes
│   └── notes/                  # meeting notes and decisions
├── references/
│   ├── papers/                 # literature PDFs/notes
│   ├── datasets/               # data source links/schema/provenance
│   └── figures/                # reference images and diagrams
├── configs/                    # dataset and experiment config files
├── notebooks/                  # EDA and exploratory analysis
├── src/citibikerl/             # implementation package
├── scripts/                    # CLI utilities and data pipeline scripts
├── outputs/
│   ├── figures/                # generated charts
│   ├── tables/                 # result tables
│   ├── models/                 # checkpoints / Q tables
│   └── logs/                   # run logs
└── tests/                      # automated checks
```

---

## Build and structure checks

```bash
make check-conflicts
make check-structure
make build-check
```

---

## Dataset-ready commands

```bash
python scripts/get_dataset.py \
  --url https://tripdata.s3.amazonaws.com/JC-202602-citibike-tripdata.csv.zip \
  --output data/raw/JC-202602-citibike-tripdata.csv
python scripts/get_weather_data.py \
  --station USW00014734 \
  --start-date 2025-01-01 \
  --end-date 2026-02-28 \
  --output data/external/noaa_daily_usw00014734_20250101_20260228.csv
make dataset-validate INPUT=data/raw/JC-202602-citibike-tripdata.csv
make preprocess-data INPUT=data/raw/JC-202602-citibike-tripdata.csv OUTPUT=data/processed/jc_202602_hourly_flows.csv
```

These commands now provide a minimal end-to-end data path: download → schema validate → preprocess hourly flows.
`get_dataset.py` extracts Citi Bike ZIP downloads when the requested output is a `.csv`, writes a per-file metadata sidecar, and updates `data/raw/_dataset_metadata.json` so multiple monthly downloads keep their provenance.
`get_weather_data.py` downloads normalized NOAA daily summaries that can be passed into experiment commands with `--weather-input`.
`configs/dataset.yaml` is now the authoritative source for raw-dataset required columns and preprocessing timezone.

---

## Training and evaluation commands

```bash
make evaluate-baseline INPUT=data/processed/jc_202602_hourly_flows.csv OUTPUT=outputs/tables/baseline_metrics.csv
make train-q-learning \
  INPUT=data/processed/jc_202602_hourly_flows.csv \
  MODEL=outputs/models/q_learning_model.json \
  TRAINING_METRICS=outputs/tables/training_metrics.csv \
  EVAL_METRICS=outputs/tables/policy_evaluation.csv
make train-dqn \
  INPUT=data/processed/jc_202602_hourly_flows.csv \
  MODEL=outputs/models/dqn_model.json \
  TRAINING_METRICS=outputs/tables/dqn_training_metrics.csv \
  EVAL_METRICS=outputs/tables/dqn_policy_evaluation.csv
make evaluate-saved-policy \
  INPUT=data/processed/jc_202602_hourly_flows.csv \
  MODEL=outputs/models/q_learning_model.json \
  OUTPUT=outputs/tables/saved_policy_evaluation.csv
make evaluate-saved-dqn \
  INPUT=data/processed/jc_202602_hourly_flows.csv \
  MODEL=outputs/models/dqn_model.json \
  OUTPUT=outputs/tables/saved_dqn_policy_evaluation.csv
make run-experiment \
  INPUT=data/processed/jc_202602_hourly_flows.csv \
  PREFIX=baseline_v1
make run-experiment \
  INPUT=data/processed/jc_202601_hourly_flows.csv,data/processed/jc_202602_hourly_flows.csv \
  PREFIX=month_holdout_v1
PYTHONPATH=src python scripts/run_experiment.py \
  --input data/processed/jc_202501_hourly_flows.csv,data/processed/jc_202502_hourly_flows.csv \
  --weather-input data/external/noaa_daily_usw00014734_20250101_20260228.csv \
  --output-prefix weather_holdout_v1
make make-plots \
  TRAINING_METRICS=outputs/tables/training_metrics.csv \
  EVAL_METRICS=outputs/tables/policy_evaluation.csv \
  REWARD_PLOT=outputs/figures/training_reward_curve.png \
  COMPARISON_PLOT=outputs/figures/policy_comparison.png
```

What the first implementation supports:
- selects the top-activity stations from the processed flow file,
- accepts one or many processed monthly CSVs via a comma-separated `INPUT`,
- simulates hourly bike inventory with no-op or bike-transfer actions,
- evaluates a **Do Not Refill / no-op** baseline,
- evaluates a training-data-driven demand-profile heuristic baseline,
- trains a tabular Q-learning agent on daily episodes using a compact forecast-aware state,
- also supports a NumPy dueling Double DQN path using dense forecast, calendar, holiday, and optional weather features,
- derives U.S. federal holiday flags for every demand day and can merge NOAA daily weather context via `--weather-input`,
- falls back to the encoded heuristic action when the Q-table encounters an unseen or low-visit forecast state,
- records `fallback_actions` and `trusted_q_actions` in tabular policy evaluation metrics so fallback-heavy runs are visible,
- uses a chronological train/test split across available days by default,
- also supports an explicit `test_start_day` cutoff for month-holdout evaluation,
- saves training metrics, evaluation metrics, and a serialized Q-table,
- reloads a saved Q-table for later evaluation,
- runs a complete experiment in one command with reproducible artifact names,
- generates reward and policy-comparison figures for the report.

Default runtime settings live in:
- `configs/environment.yaml`
- `configs/training.yaml`
- `configs/dqn_training.yaml`
- `configs/evaluation.yaml`

---

## Documentation and project operations

- Workflow commands: `docs/WORKFLOW.md`
- Execution plan and priorities: `docs/NEXT_STEPS.md`
- Weekly status board: `docs/STATUS.md`
- Dataset provenance template: `references/datasets/CITIBIKE_DATA_SOURCE.md`

---

## Starter templates included

- `docs/proposal/proposal_outline.md`
- `docs/report/report_outline.md`
- `docs/report/figure_inventory.md`
- `docs/presentation/slide_outline.md`
- `docs/presentation/asset_checklist.md`
- `docs/notes/meeting_YYYY-MM-DD_template.md`
- `docs/notes/decision_log.md`
- `docs/notes/action_items.md`
- `references/papers/reading_list.md`


If your PR still reports conflicts, follow `docs/MERGE_CONFLICT_TROUBLESHOOTING.md`.
