"""Integration tests for dataset-oriented CLI scripts."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
import zipfile


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


def test_get_dataset_extracts_zip_url_when_output_is_csv(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    output_path = tmp_path / "downloaded.csv"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "nested/raw.csv",
            "started_at,ended_at,start_station_id,start_station_name,end_station_id,end_station_name\n",
        )

    result = run_script(
        "scripts/get_dataset.py",
        "--url",
        source_zip.as_uri(),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.read_text(encoding="utf-8").startswith("started_at,ended_at")
    assert (tmp_path / "downloaded.csv.metadata.json").exists()


def test_get_dataset_aggregates_and_dedupes_metadata_index(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    with zipfile.ZipFile(first_zip, "w") as archive:
        archive.writestr("first.csv", "started_at\n2026-01-01T00:00:00Z\n")
    with zipfile.ZipFile(second_zip, "w") as archive:
        archive.writestr("second.csv", "started_at\n2026-02-01T00:00:00Z\n")

    output_first = raw_dir / "first.csv"
    output_second = raw_dir / "second.csv"

    first_run = run_script(
        "scripts/get_dataset.py",
        "--url",
        first_zip.as_uri(),
        "--output",
        str(output_first),
    )
    second_run = run_script(
        "scripts/get_dataset.py",
        "--url",
        second_zip.as_uri(),
        "--output",
        str(output_second),
    )
    third_run = run_script(
        "scripts/get_dataset.py",
        "--url",
        first_zip.as_uri(),
        "--output",
        str(output_first),
    )

    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    assert third_run.returncode == 0, third_run.stdout + third_run.stderr

    import json

    metadata_index = json.loads((raw_dir / "_dataset_metadata.json").read_text(encoding="utf-8"))
    paths = [record["path"] for record in metadata_index]
    assert paths.count(str(output_first)) == 1
    assert paths.count(str(output_second)) == 1
    assert paths[-1] == str(output_first)


def test_validate_and_preprocess_accept_zip_input(tmp_path: Path) -> None:
    raw_zip = tmp_path / "raw.zip"
    output_path = tmp_path / "processed.csv"
    config_path = tmp_path / "dataset.yaml"
    raw_csv = "\n".join(
        [
            "started_at,ended_at,start_station_id,start_station_name,end_station_id,end_station_name",
            "2026-01-01T05:30:00+00:00,2026-01-01T05:45:00+00:00,A,Alpha,B,Beta",
        ],
    )
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("raw.csv", raw_csv)

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

    validate_result = run_script(
        "scripts/validate_dataset.py",
        "--input",
        str(raw_zip),
        "--dataset-config",
        str(config_path),
    )
    preprocess_result = run_script(
        "scripts/preprocess_data.py",
        "--input",
        str(raw_zip),
        "--output",
        str(output_path),
        "--dataset-config",
        str(config_path),
    )

    assert validate_result.returncode == 0, validate_result.stdout + validate_result.stderr
    assert preprocess_result.returncode == 0, preprocess_result.stdout + preprocess_result.stderr
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    assert rows[0]["hour"] == "2026-01-01T00:00:00-05:00"
