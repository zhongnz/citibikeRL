"""Basic smoke tests for initial package build."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_import() -> None:
    import citibikerl
    import citibikerl.rebalancing

    assert hasattr(citibikerl, "__version__")
    assert hasattr(citibikerl.rebalancing, "train_q_learning")
    assert hasattr(citibikerl.rebalancing, "train_dqn")


def test_rebalancing_import_does_not_import_matplotlib() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import citibikerl.rebalancing; print('matplotlib' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "False"
