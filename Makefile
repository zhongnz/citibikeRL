.PHONY: check-conflicts verify-merge-clean check-structure build-check dataset-validate preprocess-data get-weather-data train-q-learning train-dqn evaluate-baseline evaluate-saved-policy evaluate-saved-dqn run-experiment make-plots export-presentation new-meeting new-report-draft

DATASET_CONFIG ?= configs/dataset.yaml

check-conflicts:
	./scripts/check_conflicts.sh

verify-merge-clean: check-conflicts
	git diff --check

check-structure: check-conflicts
	./scripts/check_structure.sh

build-check:
	python -m compileall src scripts
	PYTHONPATH=src python -m citibikerl.cli

dataset-validate:
	PYTHONPATH=src python scripts/validate_dataset.py --input $(INPUT) --dataset-config $(DATASET_CONFIG)

preprocess-data:
	PYTHONPATH=src python scripts/preprocess_data.py --input $(INPUT) --output $(OUTPUT) --dataset-config $(DATASET_CONFIG)

get-weather-data:
	PYTHONPATH=src python scripts/get_weather_data.py --station $(STATION) --start-date $(START_DATE) --end-date $(END_DATE) --output $(OUTPUT)

train-q-learning:
	PYTHONPATH=src python scripts/train_q_learning.py --input $(INPUT) --output-model $(MODEL) --output-training-metrics $(TRAINING_METRICS) --output-evaluation-metrics $(EVAL_METRICS)

train-dqn:
	PYTHONPATH=src python scripts/train_dqn.py --input $(INPUT) --output-model $(MODEL) --output-training-metrics $(TRAINING_METRICS) --output-evaluation-metrics $(EVAL_METRICS)

evaluate-baseline:
	PYTHONPATH=src python scripts/evaluate_baseline.py --input $(INPUT) --output $(OUTPUT)

evaluate-saved-policy:
	PYTHONPATH=src python scripts/evaluate_saved_policy.py --input $(INPUT) --model $(MODEL) --output $(OUTPUT)

evaluate-saved-dqn:
	PYTHONPATH=src python scripts/evaluate_saved_dqn.py --input $(INPUT) --model $(MODEL) --output $(OUTPUT)

run-experiment:
	PYTHONPATH=src python scripts/run_experiment.py --input $(INPUT) --output-prefix $(PREFIX)

make-plots:
	PYTHONPATH=src python scripts/make_plots.py --training-metrics $(TRAINING_METRICS) --evaluation-metrics $(EVAL_METRICS) --reward-plot $(REWARD_PLOT) --comparison-plot $(COMPARISON_PLOT)

export-presentation:
	cd docs/presentation && pandoc deck_final.md -o deck_final.pptx && pandoc deck_final.md -o deck_final.pdf

new-meeting:
	./scripts/new_meeting_note.sh

new-report-draft:
	./scripts/new_report_draft.sh $(VERSION)
