# Repository Guide

This guide explains **how to use this repository structure effectively** for a course project that includes a proposal, experiments, report, and presentation.

## A. Why this structure works

The layout follows a simple principle:

- **Inputs** (`data/raw`, references) are separated from
- **Work-in-progress analysis** (`notebooks`, `docs/notes`) and
- **Outputs** (`outputs/figures`, `outputs/tables`, `outputs/models`), while
- keeping implementation code in `src/citibikerl` and reproducible commands in `scripts`.

This separation prevents common project issues:
- mixing source files with generated files,
- losing track of report figures/tables origins,
- keeping only notebook-based logic with no migration path.

## B. Recommended naming conventions

### 1) Notebooks
Use numeric prefixes to preserve workflow order:
- `01_data_overview.ipynb`
- `02_station_sampling.ipynb`
- `03_mdp_sanity_checks.ipynb`

### 2) Outputs
Name results with date + short purpose:
- `reward_curve_2026-03-25.png`
- `baseline_comparison_2026-03-25.csv`

### 3) Docs
Store versioned docs using explicit suffixes:
- `proposal_v1.md`, `proposal_v2.md`, `proposal_final.pdf`
- `report_outline.md`, `report_draft_v1.md`, `report_final.pdf`

## C. Suggested ownership split for team collaboration

- Member A: data and EDA (`data/`, `notebooks/`, `references/datasets/`)
- Member B: method and implementation (`src/`, `scripts/`, `tests/`)
- Member C: documentation and delivery (`docs/report/`, `docs/presentation/`)

Use `docs/notes/` for meeting logs and action items.

## D. Reproducibility checklist

Before final submission:

1. Raw data source and version documented (`references/datasets/`).
2. Experiment settings centralized (`configs/`).
3. Major figures/tables reproducible from notebooks/scripts.
4. Final report and slides reference figures stored in repo.
5. README reflects the final workflow and structure.

## E. Scope boundaries for this setup stage

At the current stage, this repo includes model-free rebalancing code, experiment scripts, tests, and final report artifacts. The goal is to preserve:

- clear directory boundaries,
- clean documentation flow,
- low-friction team collaboration,
- reproducible training, evaluation, and reporting commands.
