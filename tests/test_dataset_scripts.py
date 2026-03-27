"""Integration tests for dataset-oriented CLI scripts."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a repo script with the src directory on PYTHONPATH."""
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_validate_dataset_honors_custom_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    input_path.write_text("started_at\n2026-01-01T00:15:00\n", encoding="utf-8")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "dataset:\n"
        "  required_columns:\n"
        "    - started_at\n",
        encoding="utf-8",
    )

    result = run_script(
        "scripts/validate_dataset.py",
        "--input",
        str(input_path),
        "--dataset-config",
        str(config_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_preprocess_data_applies_configured_timezone(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    config_path = tmp_path / "dataset.yaml"

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "started_at",
                "ended_at",
                "start_station_id",
                "start_station_name",
                "end_station_id",
                "end_station_name",
            ],
        )
        writer.writerow(
            [
                "2026-01-01T05:30:00+00:00",
                "2026-01-01T05:45:00+00:00",
                "A",
                "Alpha",
                "B",
                "Beta",
            ],
        )

    config_path.write_text(
        "dataset:\n"
        "  timezone: America/New_York\n"
        "  required_columns:\n"
        "    - started_at\n"
        "    - ended_at\n"
        "    - start_station_id\n"
        "    - start_station_name\n"
        "    - end_station_id\n"
        "    - end_station_name\n",
        encoding="utf-8",
    )

    result = run_script(
        "scripts/preprocess_data.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--dataset-config",
        str(config_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    assert rows == [
        {
            "hour": "2026-01-01T00:00:00-05:00",
            "start_station_id": "A",
            "end_station_id": "B",
            "trip_count": "1",
        },
    ]
