# Configs Folder

Centralize experiment settings here instead of hardcoding values in notebooks/scripts.

## Implemented config files
- `dataset.yaml` — raw and processed data defaults
- `environment.yaml` — station capacity, initial inventory, move size, reward terms
- `training.yaml` — top-N station selection, split settings, and Q-learning hyperparameters
- `dqn_training.yaml` — split settings and DQN hyperparameters
- `evaluation.yaml` — evaluation defaults for station subset selection
