"""Configuration helpers for YAML-backed scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_section(path: str | Path | None, section: str) -> dict[str, Any]:
    """Load a named section from a YAML file, returning an empty mapping when absent."""
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must define a mapping: {config_path}")

    if section in data:
        section_value = data[section]
        if section_value is None:
            return {}
        if not isinstance(section_value, dict):
            raise ValueError(f"Config section '{section}' must be a mapping: {config_path}")
        return section_value

    return data
