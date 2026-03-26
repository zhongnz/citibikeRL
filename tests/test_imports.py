"""Basic smoke tests for initial package build."""


def test_package_import() -> None:
    import citibikerl
    import citibikerl.rebalancing

    assert hasattr(citibikerl, "__version__")
    assert hasattr(citibikerl.rebalancing, "train_q_learning")
    assert hasattr(citibikerl.rebalancing, "train_dqn")
