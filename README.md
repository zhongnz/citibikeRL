# CitiBikeRL — Model-free Policy Evaluation for Small-Scale Rebalancing

This repository supports a reinforcement-learning course project on Citi Bike rebalancing.
It combines:
- **organization-first delivery** (proposal/report/presentation workflows), and
- **build-ready structure** (package skeleton, dataset scripts, validation checks).

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
python scripts/get_dataset.py --url <DATASET_URL> --output data/raw/JC-202602-citibike-tripdata.csv
make dataset-validate INPUT=data/raw/JC-202602-citibike-tripdata.csv
make preprocess-data INPUT=data/raw/JC-202602-citibike-tripdata.csv OUTPUT=data/processed/hourly_flows.csv
```

These commands now provide a minimal end-to-end data path: download → schema validate → preprocess hourly flows.

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
