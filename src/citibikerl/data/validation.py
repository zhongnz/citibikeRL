"""Dataset validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from .schema import DEFAULT_REQUIRED_COLUMNS


def missing_required_columns(
    columns: Iterable[str],
    required_columns: Iterable[str] | None = None,
) -> list[str]:
    """Return missing required schema columns."""
    present = set(columns)
    required = list(DEFAULT_REQUIRED_COLUMNS if required_columns is None else required_columns)
    return [col for col in required if col not in present]
