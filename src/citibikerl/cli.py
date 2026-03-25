"""CLI helpers for early project setup checks."""

from pathlib import Path

REQUIRED_PATHS = [
    Path("README.md"),
    Path("pyproject.toml"),
    Path("data/raw"),
    Path("data/processed"),
    Path("src/citibikerl/__init__.py"),
]


def main() -> int:
    """Validate core project layout and print status."""
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        print("Build check failed. Missing required paths:")
        for item in missing:
            print(f" - {item}")
        return 1

    print("Build check passed: core package/build files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
