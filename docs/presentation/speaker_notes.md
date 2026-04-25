# Speaker Notes

## Slide 1 — Problem
- Start with the operational intuition:
  rebalancing only helps if we move the right bikes at the right time.
- Emphasize that cost matters; otherwise every model can look good by over-moving.

## Slide 2 — Research Question
- Make clear that the real question is not whether RL beats no-op.
- The meaningful comparator is a forecast-aware heuristic.

## Slide 3 — Environment Setup
- Mention that the project deliberately simplifies the system into a five-station subproblem.
- This keeps the MDP interpretable and the experiments reproducible.

## Slide 4 — Data
- Mention that the full pipeline now uses real Citi Bike monthly data and NOAA weather.
- Note that the final evaluation is chronological, not random.

## Slide 5 — Methods
- Introduce the heuristic as a one-step forecast-based control.
- Explain that the DQN is the function-approximation extension of the earlier Q-learning path.

## Slide 6 — Development Result
- Use this slide to show why the heuristic became the main benchmark.
- The important story is that tabular RL improved with better context, but still did not surpass the heuristic.

## Slide 7 — Main Seasonal Holdout Result
- This is the core result slide.
- State directly:
  the heuristic is the strongest robust policy on the future-month holdout.
- Note that the tabular Q-table line is mostly heuristic fallback on this strict holdout, so it should not be presented as a standalone learned-policy win.

## Slide 8 — DQN Result and Caveat
- Explain the failure mode:
  unregularized DQN moved too many bikes.
- Explain the fix:
  the move-margin gate suppresses low-confidence transfers.

## Slide 9 — Robustness Check
- This slide is what keeps the claim honest.
- Say explicitly that one seed beat the heuristic, but the added seeds did not.
- The research value is the methodology and the realistic benchmark, not an overstated deep-RL win.

## Slide 10 — Takeaways
- End with the conservative conclusion:
  the heuristic is the best reliable method so far, and the DQN remains future work rather than a definitive success.
