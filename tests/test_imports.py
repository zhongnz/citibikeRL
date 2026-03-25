"""Basic smoke tests for initial package build."""


def test_package_import() -> None:
    import citibikerl

    assert hasattr(citibikerl, "__version__")
