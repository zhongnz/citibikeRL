# Decision Log

Track major project decisions with rationale.

| Date | Decision | Rationale | Impact | Owner |
|---|---|---|---|---|
| 2026-03-26 | Primary station subset fixed to `JC115`, `HB101`, `HB106`, `JC009`, `JC109` for the full-year seasonal holdout | These were the top five stations by total activity over the training window and produce a consistent, data-driven subproblem | All final experiments and writeup use the same five-station environment | team |
| 2026-03-26 | Primary evaluation uses train-through-January / test-on-February chronological holdout | This is a stronger and more defensible protocol than a random or within-week split because it exposes temporal drift | Final report should cite this as the main evaluation, not the early February prototype split | team |
| 2026-03-26 | Demand-profile heuristic retained as the main benchmark, not just no-op | The heuristic consistently outperformed no-op and most RL variants, making it the meaningful baseline | Prevents overstating RL gains against a weak comparator | team |
| 2026-03-26 | DQN move regularization is applied at policy time, not training time | Applying the move margin during training hurt learning; using it only at inference reduces gratuitous moves without constraining exploration | Final DQN claim is based on a policy-time guardrail rather than a changed reward function | team |
