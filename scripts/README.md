# Scripts Folder

Store reproducible command-line entry points.

## Available scripts
- `preprocess_data.py`
- `validate_dataset.py`
- `get_dataset.py`
- `get_weather_data.py`
- `train_q_learning.py`
- `train_dqn.py`
- `evaluate_baseline.py`
- `evaluate_saved_policy.py`
- `evaluate_saved_dqn.py`
- `run_experiment.py`
- `make_plots.py`
- `check_structure.sh`
- `check_conflicts.sh`
- `new_meeting_note.sh`
- `new_report_draft.sh`

Each script should:
- accept explicit input/output paths,
- write artifacts to `outputs/`,
- avoid hidden notebook-only logic.


## Utility
- `check_structure.sh` validates required folders/files for this scaffold.
- `new_meeting_note.sh` creates dated meeting notes from template.
- `new_report_draft.sh` creates report draft files from report outline template.

Run `make build-check` to validate imports and compile Python sources.

Use `make dataset-validate INPUT=<raw_csv>` before preprocessing.

Use `make get-weather-data STATION=<noaa_station> START_DATE=<yyyy-mm-dd> END_DATE=<yyyy-mm-dd> OUTPUT=<weather_csv>` to fetch normalized NOAA daily weather summaries.

Use `make evaluate-baseline INPUT=<processed_csv_or_csvs> OUTPUT=<metrics_csv>` to benchmark the no-op baseline.

Use `make train-q-learning INPUT=<processed_csv_or_csvs> MODEL=<model_json> TRAINING_METRICS=<training_csv> EVAL_METRICS=<evaluation_csv>` to train and evaluate the tabular agent.
By default this now trains on the earliest `train_fraction` share of days and evaluates on the held-out tail days.
You can also pass `--test-start-day YYYY-MM-DD` for an explicit month-boundary holdout.
The learned Q-table now uses a compact forecast-aware state derived from the training split's demand profile, U.S. federal holiday flags, and optional NOAA daily weather context via `--weather-input`.
It falls back to the encoded heuristic action on unseen states.
The learned policy can also require a minimum historical visit count before trusting the Q-table; the default config now uses a conservative `min_state_visit_count`.
The evaluation CSV includes `baseline_no_op`, `heuristic_demand_profile`, and `trained_q_policy`.

Use `make train-dqn INPUT=<processed_csv_or_csvs> MODEL=<model_json> TRAINING_METRICS=<training_csv> EVAL_METRICS=<evaluation_csv>` to train a dueling Double DQN-style agent implemented in NumPy.
This path uses dense forecast, calendar, holiday, and optional weather features instead of a tabular state dictionary.
The evaluation CSV includes `baseline_no_op`, `heuristic_demand_profile`, and `trained_dqn_policy`.

Use `make evaluate-saved-policy INPUT=<processed_csv_or_csvs> MODEL=<model_json> OUTPUT=<metrics_csv>` to re-evaluate a serialized policy on the same station subset it was trained on.

Use `make evaluate-saved-dqn INPUT=<processed_csv_or_csvs> MODEL=<model_json> OUTPUT=<metrics_csv>` to re-evaluate a serialized DQN policy.

Use `make run-experiment INPUT=<processed_csv_or_csvs> PREFIX=<name>` to produce a complete set of models, tables, figures, and summary metadata under the standard `outputs/` folders.

Use `make make-plots TRAINING_METRICS=<training_csv> EVAL_METRICS=<evaluation_csv> REWARD_PLOT=<png> COMPARISON_PLOT=<png>` to create report-ready figures.

- `check_conflicts.sh` fails if merge conflict markers are found.
