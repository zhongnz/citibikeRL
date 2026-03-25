# Next Steps (Execution Plan)

This is the recommended sequence to move from scaffold to a complete course project.

## Priority 0 — Team alignment (Day 1)

1. Finalize station sample scope (which 5 stations and why).
2. Confirm project success metrics:
   - served trips,
   - unmet demand,
   - cumulative reward,
   - refill action count.
3. Assign owners for:
   - data + EDA,
   - environment + baseline,
   - report + slides.

Deliverables:
- Update `docs/notes/decision_log.md`.
- Create one meeting note with `make new-meeting`.

---

## Priority 1 — Data readiness (Days 1–3)

1. Download the target Citi Bike file into `data/raw/`.
2. Record provenance in `references/datasets/CITIBIKE_DATA_SOURCE.md`.
3. Create one EDA notebook (`notebooks/01_data_overview.ipynb`) with:
   - key columns sanity checks,
   - hourly demand overview,
   - station frequency summary.

Deliverables:
- Data provenance entry completed.
- One EDA notebook checked in.

---

## Priority 2 — Environment spec freeze (Days 3–5)

1. Freeze MDP choices in proposal/report outlines:
   - state variables,
   - action space,
   - transition order,
   - reward terms.
2. Lock the baseline definition (**Do Not Refill**).
3. Confirm evaluation protocol (episode horizon, number of runs, comparison method).

Deliverables:
- Updated `docs/proposal/proposal_outline.md` with finalized assumptions.
- Updated `docs/report/report_outline.md` with final methods/evaluation wording.

---

## Priority 3 — Minimal implementation start (Week 2)

1. Implement data preprocessing script.
2. Implement environment simulation.
3. Implement baseline policy rollout.
4. Implement tabular Q-learning training loop.

Deliverables:
- First runnable end-to-end pipeline producing metrics into `outputs/tables/`.

---

## Priority 4 — Results and analysis (Week 2–3)

1. Run baseline and Q-learning comparisons.
2. Generate core figures into `outputs/figures/`.
3. Fill `docs/report/figure_inventory.md` with source paths.

Deliverables:
- Initial result table + at least 2 figures.

---

## Priority 5 — Final report and presentation (Final week)

1. Draft and finalize `docs/report/report_draft_vX.md`.
2. Build final slide deck from `docs/presentation/slide_outline.md`.
3. Complete `docs/presentation/asset_checklist.md`.

Deliverables:
- Final report PDF.
- Final presentation deck + speaker notes.

---

## Immediate 3-task recommendation (start now)

If you only do three things next, do these first:

1. Add the real dataset file and provenance entry.
2. Create `notebooks/01_data_overview.ipynb`.
3. Finalize the exact 5-station sample decision in `docs/notes/decision_log.md`.
