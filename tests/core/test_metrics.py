"""
Unit tests for Metric, MetricCollection, built-in metrics, and
metric registry.

Covers: update/compute/reset/merge lifecycle for all 4 built-in
metrics, MetricCollection add/remove/update/compute/reset/clear,
duplicate metric, merge wrong type, compute without update,
registry integration, deterministic ordering, state property.
"""

import pytest

from bio_oncograph.core.metrics import (
    CountMetric,
    MeanMetric,
    Metric,
    MetricCollection,
    RunningAverageMetric,
    SumMetric,
    create_metric,
    list_registered_metrics,
    register_metric,
)


# ══════════════════════════════════════════════════════════════════
# MeanMetric
# ══════════════════════════════════════════════════════════════════


def test_mean_update_compute() -> None:
    """Mean of [2, 4, 6] is 4.0."""
    m = MeanMetric()
    m.update(2.0, None)
    m.update(4.0, None)
    m.update(6.0, None)
    assert m.compute() == pytest.approx(4.0)


def test_mean_reset() -> None:
    """Reset clears accumulated state."""
    m = MeanMetric()
    m.update(10.0, None)
    m.reset()
    assert m.compute() == 0.0


def test_mean_merge() -> None:
    """Merge combines two MeanMetrics."""
    a = MeanMetric()
    a.update(2.0, None)
    a.update(4.0, None)
    b = MeanMetric()
    b.update(6.0, None)
    a.merge(b)
    assert a.compute() == pytest.approx(4.0)  # (2+4+6)/3


def test_mean_merge_wrong_type_raises() -> None:
    """Merging incompatible types raises TypeError."""
    m = MeanMetric()
    with pytest.raises(TypeError, match="Cannot merge"):
        m.merge(SumMetric())


def test_mean_compute_without_update() -> None:
    """compute() returns 0.0 with no updates."""
    assert MeanMetric().compute() == 0.0


def test_mean_state() -> None:
    """State dict contains sum and count."""
    m = MeanMetric()
    m.update(5.0, None)
    s = m.state
    assert s["sum"] == 5.0
    assert s["count"] == 1


def test_mean_name() -> None:
    """Custom name is exposed."""
    m = MeanMetric(metric_name="loss")
    assert m.name == "loss"


# ══════════════════════════════════════════════════════════════════
# SumMetric
# ══════════════════════════════════════════════════════════════════


def test_sum_update_compute() -> None:
    """Sum of [1, 2, 3] is 6.0."""
    m = SumMetric()
    m.update(1.0, None)
    m.update(2.0, None)
    m.update(3.0, None)
    assert m.compute() == pytest.approx(6.0)


def test_sum_reset() -> None:
    """Reset clears the total."""
    m = SumMetric()
    m.update(100.0, None)
    m.reset()
    assert m.compute() == 0.0


def test_sum_merge() -> None:
    """Merge adds totals."""
    a = SumMetric()
    a.update(10.0, None)
    b = SumMetric()
    b.update(20.0, None)
    a.merge(b)
    assert a.compute() == pytest.approx(30.0)


def test_sum_merge_wrong_type() -> None:
    """Merging wrong type raises TypeError."""
    with pytest.raises(TypeError):
        SumMetric().merge(MeanMetric())


# ══════════════════════════════════════════════════════════════════
# CountMetric
# ══════════════════════════════════════════════════════════════════


def test_count_update_compute() -> None:
    """Count after 5 updates is 5."""
    m = CountMetric()
    for _ in range(5):
        m.update(None, None)
    assert m.compute() == 5.0


def test_count_reset() -> None:
    """Reset clears count."""
    m = CountMetric()
    m.update(None, None)
    m.reset()
    assert m.compute() == 0.0


def test_count_merge() -> None:
    """Merge adds counts."""
    a = CountMetric()
    a.update(None, None)
    a.update(None, None)
    b = CountMetric()
    b.update(None, None)
    a.merge(b)
    assert a.compute() == 3.0


def test_count_merge_wrong_type() -> None:
    """Merging wrong type raises TypeError."""
    with pytest.raises(TypeError):
        CountMetric().merge(SumMetric())


# ══════════════════════════════════════════════════════════════════
# RunningAverageMetric
# ══════════════════════════════════════════════════════════════════


def test_ema_first_update() -> None:
    """First update sets the value directly."""
    m = RunningAverageMetric(alpha=0.5)
    m.update(10.0, None)
    assert m.compute() == pytest.approx(10.0)


def test_ema_subsequent_update() -> None:
    """EMA formula: alpha * new + (1-alpha) * old."""
    m = RunningAverageMetric(alpha=0.5)
    m.update(10.0, None)
    m.update(20.0, None)
    # 0.5*20 + 0.5*10 = 15.0
    assert m.compute() == pytest.approx(15.0)


def test_ema_reset() -> None:
    """Reset clears value and count."""
    m = RunningAverageMetric()
    m.update(5.0, None)
    m.reset()
    assert m.compute() == 0.0


def test_ema_invalid_alpha() -> None:
    """alpha outside (0,1] raises ValueError."""
    with pytest.raises(ValueError, match="alpha"):
        RunningAverageMetric(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        RunningAverageMetric(alpha=1.5)


def test_ema_merge() -> None:
    """Merge computes weighted average."""
    a = RunningAverageMetric(alpha=0.5)
    a.update(10.0, None)
    a.update(20.0, None)  # value=15, count=2
    b = RunningAverageMetric(alpha=0.5)
    b.update(30.0, None)  # value=30, count=1
    a.merge(b)
    expected = (15.0 * 2 + 30.0 * 1) / 3
    assert a.compute() == pytest.approx(expected)


def test_ema_merge_wrong_type() -> None:
    """Merging wrong type raises TypeError."""
    with pytest.raises(TypeError):
        RunningAverageMetric().merge(MeanMetric())


# ══════════════════════════════════════════════════════════════════
# MetricCollection
# ══════════════════════════════════════════════════════════════════


def test_collection_add_and_list() -> None:
    """add() inserts; list_metrics() returns names in order."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="a"))
    mc.add(SumMetric(metric_name="b"))
    assert mc.list_metrics() == ["a", "b"]


def test_collection_len() -> None:
    """len() returns metric count."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="x"))
    assert len(mc) == 1


def test_collection_contains() -> None:
    """'in' operator works."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="x"))
    assert "x" in mc
    assert "y" not in mc


def test_collection_duplicate_raises() -> None:
    """Adding a metric with the same name raises ValueError."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="dup"))
    with pytest.raises(ValueError, match="already exists"):
        mc.add(SumMetric(metric_name="dup"))


def test_collection_remove() -> None:
    """remove() deletes by name."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="r"))
    mc.remove("r")
    assert "r" not in mc


def test_collection_remove_missing_raises() -> None:
    """Removing unknown name raises KeyError."""
    mc = MetricCollection()
    with pytest.raises(KeyError):
        mc.remove("nonexistent")


def test_collection_update_compute() -> None:
    """update+compute forwards to all metrics."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="mean"))
    mc.add(CountMetric(metric_name="count"))
    mc.update(4.0, None)
    mc.update(6.0, None)
    result = mc.compute()
    assert result["mean"] == pytest.approx(5.0)
    assert result["count"] == pytest.approx(2.0)


def test_collection_reset() -> None:
    """reset clears all metrics."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="m"))
    mc.update(10.0, None)
    mc.reset()
    assert mc.compute()["m"] == 0.0


def test_collection_clear() -> None:
    """clear removes all metrics."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="m"))
    mc.clear()
    assert len(mc) == 0


def test_collection_deterministic_ordering() -> None:
    """compute returns metrics in insertion order."""
    mc = MetricCollection()
    mc.add(SumMetric(metric_name="z"))
    mc.add(MeanMetric(metric_name="a"))
    mc.add(CountMetric(metric_name="m"))
    mc.update(1.0, None)
    keys = list(mc.compute().keys())
    assert keys == ["z", "a", "m"]


def test_collection_get() -> None:
    """get() retrieves by name."""
    mc = MetricCollection()
    m = MeanMetric(metric_name="x")
    mc.add(m)
    assert mc.get("x") is m


def test_collection_get_missing_raises() -> None:
    """get() unknown name raises KeyError."""
    mc = MetricCollection()
    with pytest.raises(KeyError):
        mc.get("missing")


def test_collection_non_metric_raises() -> None:
    """add() with non-Metric raises TypeError."""
    mc = MetricCollection()
    with pytest.raises(TypeError, match="Expected Metric"):
        mc.add("not_a_metric")  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════
# Metric Registry
# ══════════════════════════════════════════════════════════════════


def test_list_registered_includes_builtins() -> None:
    """Built-in metrics are pre-registered."""
    names = list_registered_metrics()
    assert "mean" in names
    assert "sum" in names
    assert "count" in names
    assert "running_avg" in names


def test_create_metric_from_registry() -> None:
    """create_metric returns correct type."""
    m = create_metric("mean", metric_name="my_mean")
    assert isinstance(m, MeanMetric)
    assert m.name == "my_mean"


def test_create_metric_unknown_raises() -> None:
    """Unknown name raises KeyError."""
    with pytest.raises(KeyError):
        create_metric("nonexistent_metric_xyz")
