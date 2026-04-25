"""CSV file opening helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import io
from pathlib import Path
from typing import TextIO
import zipfile


@contextmanager
def open_csv_text(input_path: str | Path) -> Iterator[TextIO]:
    """Open a plain CSV or the first CSV member of a ZIP archive as text."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV input not found: {path}")

    if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            csv_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            ]
            if not csv_members:
                raise ValueError(f"ZIP archive does not contain a CSV file: {path}")

            with archive.open(csv_members[0], "r") as raw_handle:
                text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
                try:
                    yield text_handle
                finally:
                    text_handle.close()
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        yield handle
