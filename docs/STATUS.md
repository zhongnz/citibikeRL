# Project Status Board

Use this file as a single source of truth for weekly progress.

## Milestones

- [x] M0: Repo scaffold finalized
- [x] M1: Dataset downloaded + provenance documented
- [x] M2: 5-station sample finalized
- [x] M3: EDA notebook completed
- [x] M4: Environment + baseline implemented
- [x] M5: Q-learning training and evaluation completed
- [x] M6: Report draft complete
- [ ] M7: Final report + presentation submitted

---

## Current week focus

- Week of: 2026-03-23
- Goal:
  finish the research narrative around the full-year train / February holdout and prepare report-slide assets.
- Expected deliverables:
  DQN robustness check, report draft, figure inventory, and updated slide outline.

---

## Owner tracking

| Workstream | Owner | Status | Notes |
|---|---|---|---|
| Data & preprocessing | team | Done | Citi Bike and NOAA data ingested with provenance |
| Environment & baseline | team | Done | No-op and demand-profile heuristic implemented |
| Training & evaluation | team | In review | Seasonal holdout complete; DQN seed stability still weak |
| Report writing | team | In progress | Draft v1 and figure inventory prepared |
| Presentation | team | In progress | Slide outline ready; deck not exported |

---

## Risks (active)

| Risk | Severity | Mitigation | Owner |
|---|---|---|---|
| DQN improvement is not stable across seeds | High | Report the seed sweep explicitly; present the heuristic as the strongest robust baseline | team |
| Final claim could overstate one strong DQN run | High | Center the writeup on robustness, not best-case reward | team |
| Presentation assets are still not assembled into a deck | Medium | Convert slide outline into deck and export PDF/PPTX | team |
