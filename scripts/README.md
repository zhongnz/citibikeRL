# Scripts Folder

Store reproducible command-line entry points.

## Planned scripts
- `preprocess_data.py`
- `validate_dataset.py`
- `get_dataset.py`
- `train_q_learning.py`
- `evaluate_baseline.py`
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

- `check_conflicts.sh` fails if merge conflict markers are found.
