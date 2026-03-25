# CitiBikeRL — Repository Setup for RL Project Delivery

This repository is now structured for a reinforcement learning course project on Citi Bike rebalancing, with emphasis on **organization first** (proposal, report, presentation, reproducibility), before implementation code.

## Project objective (current stage)

Create a clean, team-friendly workspace that supports:
- proposal development,
- data provenance tracking,
- experiment-ready folder boundaries,
- final report and presentation production.

---

## Repository map

```text
citibikeRL/
├── README.md
├── .gitignore
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
├── configs/                    # future yaml/json config files
├── notebooks/                  # EDA and exploratory analysis
├── src/citibikerl/             # future implementation package
├── scripts/                    # future reproducible CLI scripts
├── outputs/
│   ├── figures/                # generated charts
│   ├── tables/                 # result tables
│   ├── models/                 # checkpoints / Q tables
│   └── logs/                   # run logs
└── tests/                      # future automated checks
```

---

## What to put where (practical)

- Put original Citi Bike downloads in `data/raw/` and do not edit them.
- Put derived datasets in `data/processed/`.
- Track source URLs and schema notes in `references/datasets/`.
- Keep proposal/report/slides in `docs/` subfolders (with versioned filenames).
- Keep generated plots and tables in `outputs/` so report assets are traceable.

---

## Team delivery workflow

1. **Proposal phase**
   - Use `docs/proposal/README.md` checklist.

2. **Planning and coordination**
   - Record decisions in `docs/notes/`.

3. **Data documentation phase**
   - Fill `references/datasets/CITIBIKE_DATA_SOURCE.md`.

4. **Experiment phase (later)**
   - Keep reusable logic in `src/`, runner scripts in `scripts/`.

5. **Final delivery phase**
   - Report assets in `docs/report/`.
   - Presentation assets in `docs/presentation/`.

---


## Starter templates included

To make the repo immediately usable, starter templates are already provided:
- `docs/proposal/proposal_outline.md`
- `docs/report/report_outline.md`
- `docs/report/figure_inventory.md`
- `docs/presentation/slide_outline.md`
- `docs/presentation/asset_checklist.md`
- `docs/notes/meeting_YYYY-MM-DD_template.md`
- `docs/notes/decision_log.md`
- `docs/notes/action_items.md`
- `references/papers/reading_list.md`

---


## Quick validation command

Run this any time to confirm the scaffold is intact:

```bash
make check-structure
```

---

## Immediate next actions

- Add your proposal document to `docs/proposal/`.
- Add at least one dataset provenance entry in `references/datasets/CITIBIKE_DATA_SOURCE.md`.
- Create the first meeting note in `docs/notes/`.
- Draft report outline in `docs/report/` and slide outline in `docs/presentation/`.

This keeps the repo simple, consistent, and ready for implementation when you start coding.


## Operations cheatsheet

See `docs/WORKFLOW.md` for day-to-day commands (validate scaffold, create meeting notes, create report drafts).

For the execution sequence, see `docs/NEXT_STEPS.md`.

Track milestone progress in `docs/STATUS.md`.


## Build started (v0.1)

The repository now includes an initial Python package build scaffold:
- `pyproject.toml`
- `src/citibikerl/` package with version + CLI check
- `scripts/preprocess_data.py` starter entry point

Quick checks:
```bash
make check-structure
make build-check
```



## Dataset-ready commands

Once you have a dataset URL:

```bash
python scripts/get_dataset.py --url <DATASET_URL> --output data/raw/JC-202602-citibike-tripdata.csv
make dataset-validate INPUT=data/raw/JC-202602-citibike-tripdata.csv
make preprocess-data INPUT=data/raw/JC-202602-citibike-tripdata.csv OUTPUT=data/processed/hourly_flows.csv
```

