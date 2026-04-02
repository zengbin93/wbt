import wbt


def test_public_exports() -> None:
    assert wbt.WeightBacktest is not None
    assert wbt.daily_performance is not None
