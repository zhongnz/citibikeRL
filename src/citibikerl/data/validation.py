"""Dataset validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from .schema import REQUIRED_COLUMNS


def missing_required_columns(columns: Iterable[str]) -> list[str]:
    """Return missing required schema columns."""
    present = set(columns)
    return [col for col in REQUIRED_COLUMNS if col not in present]
