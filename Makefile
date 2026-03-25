.PHONY: check-conflicts verify-merge-clean check-structure build-check dataset-validate preprocess-data new-meeting new-report-draft

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
	PYTHONPATH=src python scripts/validate_dataset.py --input $(INPUT)

preprocess-data:
	PYTHONPATH=src python scripts/preprocess_data.py --input $(INPUT) --output $(OUTPUT)

new-meeting:
	./scripts/new_meeting_note.sh

new-report-draft:
	./scripts/new_report_draft.sh $(VERSION)
