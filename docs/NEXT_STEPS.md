# Next Steps

The project is no longer in scaffold mode. The remaining work is final-mile research packaging and a small amount of robustness follow-up.

## What is done

1. Citi Bike demand and NOAA weather data are ingested with provenance.
2. The five-station environment, no-op baseline, heuristic baseline, tabular Q-learning path, and DQN path are implemented.
3. The primary evaluation protocol is fixed:
   train on January 2025 through January 2026 and test on February 2026.
4. Report-ready figures and a draft report now exist.

## What remains

### Priority 1 — Finalize the written claim

1. Convert `docs/report/report_draft_v1.md` into the final report.
2. Keep the main claim conservative:
   the heuristic is the strongest robust policy, while the regularized DQN is promising but unstable.
3. Pull final figure references from `docs/report/figure_inventory.md`.

Deliverables:

- Final report PDF.

### Priority 2 — Finish the presentation

1. Turn `docs/presentation/slide_outline.md` into the actual deck.
2. Add the seed-sweep robustness table to the slides.
3. Export PDF and PPTX and complete `docs/presentation/asset_checklist.md`.

Deliverables:

- Final slide deck.
- Speaker notes.

### Priority 3 — Optional but valuable robustness work

1. Run more DQN seeds on the final regularized setup.
2. If time permits, test residual learning around the heuristic baseline.
3. If a stronger claim is needed, hold out an additional future month rather than tuning further on February.

Deliverables:

- Expanded robustness appendix or extra table.

## Immediate 3-task recommendation

1. Build the final slide deck from the updated slide outline.
2. Export the report draft to PDF after one editing pass.
3. Decide whether the team wants to claim "best run" or "robust winner"; the current evidence supports only the latter for the heuristic.
