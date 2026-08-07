"""
Unit tests for Evaluator.

Covers: evaluate_batch, evaluate_epoch, summary, reset,
multi-epoch workflow, empty evaluator, epoch auto-reset,
metrics property, and num_epochs property.
"""

import pytest

from bio_oncograph.core.evaluator import Evaluator
from bio_oncograph.core.metrics import (
    CountMetric,
    MeanMetric,
    MetricCollection,
    SumMetric,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def collection() -> MetricCollection:
    """MetricCollection with mean and count."""
    mc = MetricCollection()
    mc.add(MeanMetric(metric_name="mean_loss"))
    mc.add(CountMetric(metric_name="num_batches"))
    return mc


@pytest.fixture()
def evaluator(collection: MetricCollection) -> Evaluator:
    """Evaluator wrapping the fixture collection."""
    return Evaluator(collection)


# ── evaluate_batch ──────────────────────────────────────────────────


def test_evaluate_batch_updates_metrics(
    evaluator: Evaluator,
) -> None:
    """evaluate_batch forwards to all metrics."""
    evaluator.evaluate_batch(4.0, None)
    evaluator.evaluate_batch(6.0, None)
    result = evaluator.metrics.compute()
    assert result["mean_loss"] == pytest.approx(5.0)
    assert result["num_batches"] == pytest.approx(2.0)


# ── evaluate_epoch ──────────────────────────────────────────────────


def test_evaluate_epoch_computes_and_resets(
    evaluator: Evaluator,
) -> None:
    """evaluate_epoch returns metrics and resets for next epoch."""
    evaluator.evaluate_batch(10.0, None)
    result = evaluator.evaluate_epoch()
    assert result["mean_loss"] == pytest.approx(10.0)
    # After epoch, metrics are reset
    assert evaluator.metrics.compute()["mean_loss"] == 0.0


def test_evaluate_epoch_stores_result(
    evaluator: Evaluator,
) -> None:
    """evaluate_epoch appends to internal history."""
    evaluator.evaluate_batch(5.0, None)
    evaluator.evaluate_epoch()
    assert evaluator.num_epochs == 1


# ── Multi-epoch workflow ────────────────────────────────────────────


def test_multi_epoch_workflow(
    evaluator: Evaluator,
) -> None:
    """Two epochs accumulate independently."""
    # Epoch 0
    evaluator.evaluate_batch(2.0, None)
    evaluator.evaluate_batch(4.0, None)
    e0 = evaluator.evaluate_epoch()
    assert e0["mean_loss"] == pytest.approx(3.0)

    # Epoch 1
    evaluator.evaluate_batch(10.0, None)
    e1 = evaluator.evaluate_epoch()
    assert e1["mean_loss"] == pytest.approx(10.0)

    assert evaluator.num_epochs == 2


# ── summary ─────────────────────────────────────────────────────────


def test_summary_after_epochs(
    evaluator: Evaluator,
) -> None:
    """summary contains all epoch results."""
    evaluator.evaluate_batch(1.0, None)
    evaluator.evaluate_epoch()
    evaluator.evaluate_batch(2.0, None)
    evaluator.evaluate_epoch()

    s = evaluator.summary()
    assert s["num_epochs"] == 2
    assert len(s["epochs"]) == 2
    assert s["latest"]["mean_loss"] == pytest.approx(2.0)


def test_summary_empty(evaluator: Evaluator) -> None:
    """summary with no epochs returns empty latest."""
    s = evaluator.summary()
    assert s["num_epochs"] == 0
    assert s["epochs"] == []
    assert s["latest"] == {}


# ── reset ───────────────────────────────────────────────────────────


def test_reset_clears_everything(
    evaluator: Evaluator,
) -> None:
    """reset clears metrics and epoch history."""
    evaluator.evaluate_batch(5.0, None)
    evaluator.evaluate_epoch()
    evaluator.reset()
    assert evaluator.num_epochs == 0
    s = evaluator.summary()
    assert s["num_epochs"] == 0
    assert s["latest"] == {}


# ── Properties ──────────────────────────────────────────────────────


def test_metrics_property(
    evaluator: Evaluator, collection: MetricCollection
) -> None:
    """metrics property returns the collection."""
    assert evaluator.metrics is collection


def test_num_epochs_initial(evaluator: Evaluator) -> None:
    """num_epochs starts at 0."""
    assert evaluator.num_epochs == 0
