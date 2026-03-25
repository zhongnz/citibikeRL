# CitiBikeRL — Model-free Policy Evaluation for Small-Scale Rebalancing

This repository is organized to support a reinforcement learning course project on **bike-sharing rebalancing** using Citi Bike historical trip data.

At this stage, the repo focuses on **project setup and research workflow** (not implementation code yet), so the structure supports:

- proposal-to-experiment traceability,
- clean data management,
- report and presentation production,
- future reproducible RL experiments.

## 1) Project Overview

### Problem
Bike-share systems experience inventory imbalance: some stations run out of bikes while others become overfull. This can reduce served trips and increase unmet demand.

### Core idea
Model hourly rebalancing decisions as a Markov Decision Process (MDP), evaluate a tabular Q-learning policy, and compare against a **Do Not Refill** baseline.

### Why this repo layout
The folder structure is intentionally simple and course-friendly: it separates raw assets, generated artifacts, documentation, and future code.

---

## 2) Repository Structure

```text
citibikeRL/
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   ├── proposal/
│   ├── report/
│   ├── presentation/
│   └── notes/
├── references/
│   ├── papers/
│   ├── datasets/
│   └── figures/
├── configs/
├── notebooks/
├── src/
│   └── citibikerl/
├── scripts/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── logs/
└── tests/
```

---

## 3) Folder-by-folder Description

### `data/`
- `raw/`: immutable source data (original downloads, unmodified).
- `processed/`: cleaned/aggregated data used by experiments.
- `external/`: auxiliary third-party files (metadata, maps, etc.).

### `docs/`
- `proposal/`: project proposal drafts/final version.
- `report/`: report outline, drafts, and final report.
- `presentation/`: slide deck, speaker notes, backup slides.
- `notes/`: meeting notes, decisions, TODO logs.

### `references/`
- `papers/`: literature PDFs and reading notes.
- `datasets/`: source links, schema notes, citations.
- `figures/`: reference images/diagrams used for report or slides.

### `configs/`
Reserved for future experiment configuration files (station set, reward weights, training hyperparameters).

### `notebooks/`
Exploratory analysis, quick visual diagnostics, and early validation.

### `src/citibikerl/`
Future implementation package (environment, agents, baselines, evaluation modules).

### `scripts/`
Future reproducible command-line entry points (preprocess/train/evaluate/plot).

### `outputs/`
Generated artifacts from experiments:
- `figures/`: plots for report/slides,
- `tables/`: result summaries,
- `models/`: serialized policies/checkpoints,
- `logs/`: run logs and diagnostics.

### `tests/`
Future automated checks for data transforms, environment logic, and metrics correctness.

---

## 4) Data and Versioning Policy (Important)

1. Keep `data/raw/` immutable (never overwrite source files).
2. Treat `data/processed/` and `outputs/` as reproducible artifacts.
3. Do not commit very large datasets or model artifacts to Git unless explicitly needed.
4. Record data source links and schema notes under `references/datasets/`.

---

## 5) Suggested Team Workflow (Proposal → Report → Presentation)

1. **Proposal phase**
   - Keep proposal materials in `docs/proposal/`.
   - Keep background reading in `references/papers/`.

2. **Data understanding phase**
   - Store raw trip data in `data/raw/`.
   - Use `notebooks/` for EDA and station selection reasoning.

3. **Experiment phase (later)**
   - Put reusable code in `src/citibikerl/`.
   - Use `scripts/` for reproducible runs.
   - Save generated artifacts in `outputs/`.

4. **Report & presentation phase**
   - Draft and finalize report in `docs/report/`.
   - Build presentation materials in `docs/presentation/`.
   - Export final figures/tables from `outputs/` to docs as needed.

---

## 6) Minimal Next Steps

- Add the finalized proposal PDF/markdown to `docs/proposal/`.
- Add dataset source note to `references/datasets/`.
- Add station sample rationale note to `docs/notes/`.
- Start a first EDA notebook under `notebooks/`.

This keeps setup simple while preserving a clean path to implementation and final deliverables.
