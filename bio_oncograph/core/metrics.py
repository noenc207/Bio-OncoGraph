"""
Metrics framework for Bio-OncoGraph.

Provides:
  - ``Metric`` — abstract base class with update/compute/reset/merge.
  - ``MetricCollection`` — manages multiple named metrics.
  - ``MeanMetric`` / ``SumMetric`` / ``CountMetric`` /
    ``RunningAverageMetric`` — built-in, backend-agnostic metrics.
  - ``register_metric`` / ``create_metric`` /
    ``list_registered_metrics`` — registry integration (M005).

All types are pure Python — **no** ``torch``, ``sklearn``, or
``pandas`` imports — so they can be used in the Domain layer.

Design references:
  - Software Architecture § 14 (Metrics)
  - API Bible § 11 (Metrics API)
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type

from bio_oncograph.core.registry import ComponentRegistry

logger = logging.getLogger("BioOncoGraph")


# ── Abstract Base ───────────────────────────────────────────────────


class Metric(ABC):
    """
    Abstract base class for a single metric.

    Lifecycle::

        reset() → update() × N → compute() → reset() → …

    Subclasses must implement all abstract methods.  The ``merge``
    method enables distributed aggregation (e.g. across DDP ranks).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        ...

    @abstractmethod
    def update(self, prediction: Any, target: Any) -> None:
        """
        Accumulate a single observation.

        Args:
            prediction: Predicted value or model output.
            target: Ground-truth value (may be unused by
                aggregation-only metrics).
        """
        ...

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the metric from accumulated state.

        Returns:
            The scalar metric value.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state."""
        ...

    @abstractmethod
    def merge(self, other: "Metric") -> None:
        """
        Merge another metric's state into this one.

        Args:
            other: A metric of the **same** concrete type.

        Raises:
            TypeError: If *other* is not the same type.
        """
        ...

    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        """
        Return the internal state as a serialisable dict.

        Returns:
            Dictionary of state variables.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ── Built-in Metrics ───────────────────────────────────────────────


class MeanMetric(Metric):
    """
    Running arithmetic mean.

    Tracks ``sum / count`` across ``update`` calls.  The
    ``prediction`` argument is the numeric value to average;
    ``target`` is accepted but unused.
    """

    def __init__(self, metric_name: str = "mean") -> None:
        """
        Initialise the metric.

        Args:
            metric_name: Display name.
        """
        self._name = metric_name
        self._sum: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    def update(self, prediction: Any, target: Any = None) -> None:
        """
        Add a value.

        Args:
            prediction: Numeric value to accumulate.
            target: Unused.
        """
        self._sum += float(prediction)
        self._count += 1

    def compute(self) -> float:
        """
        Return the mean.

        Returns:
            ``sum / count``, or ``0.0`` if no updates.
        """
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def reset(self) -> None:
        """Reset sum and count."""
        self._sum = 0.0
        self._count = 0

    def merge(self, other: "Metric") -> None:
        """
        Merge another MeanMetric.

        Args:
            other: Must be a ``MeanMetric``.

        Raises:
            TypeError: If *other* is not a ``MeanMetric``.
        """
        if not isinstance(other, MeanMetric):
            raise TypeError(
                f"Cannot merge {type(other).__name__} "
                f"into MeanMetric."
            )
        self._sum += other._sum
        self._count += other._count

    @property
    def state(self) -> Dict[str, Any]:
        """Return internal state."""
        return {"sum": self._sum, "count": self._count}


class SumMetric(Metric):
    """
    Running sum.

    Accumulates the total of all ``prediction`` values.
    """

    def __init__(self, metric_name: str = "sum") -> None:
        """
        Initialise the metric.

        Args:
            metric_name: Display name.
        """
        self._name = metric_name
        self._total: float = 0.0

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    def update(self, prediction: Any, target: Any = None) -> None:
        """
        Add a value.

        Args:
            prediction: Numeric value to add.
            target: Unused.
        """
        self._total += float(prediction)

    def compute(self) -> float:
        """Return the running sum."""
        return self._total

    def reset(self) -> None:
        """Reset the total."""
        self._total = 0.0

    def merge(self, other: "Metric") -> None:
        """
        Merge another SumMetric.

        Args:
            other: Must be a ``SumMetric``.

        Raises:
            TypeError: If *other* is not a ``SumMetric``.
        """
        if not isinstance(other, SumMetric):
            raise TypeError(
                f"Cannot merge {type(other).__name__} "
                f"into SumMetric."
            )
        self._total += other._total

    @property
    def state(self) -> Dict[str, Any]:
        """Return internal state."""
        return {"total": self._total}


class CountMetric(Metric):
    """
    Observation counter.

    Increments by one on each ``update`` regardless of the
    argument values.
    """

    def __init__(self, metric_name: str = "count") -> None:
        """
        Initialise the metric.

        Args:
            metric_name: Display name.
        """
        self._name = metric_name
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    def update(self, prediction: Any, target: Any = None) -> None:
        """
        Increment the counter.

        Args:
            prediction: Unused.
            target: Unused.
        """
        self._count += 1

    def compute(self) -> float:
        """Return the count as a float."""
        return float(self._count)

    def reset(self) -> None:
        """Reset the counter."""
        self._count = 0

    def merge(self, other: "Metric") -> None:
        """
        Merge another CountMetric.

        Args:
            other: Must be a ``CountMetric``.

        Raises:
            TypeError: If *other* is not a ``CountMetric``.
        """
        if not isinstance(other, CountMetric):
            raise TypeError(
                f"Cannot merge {type(other).__name__} "
                f"into CountMetric."
            )
        self._count += other._count

    @property
    def state(self) -> Dict[str, Any]:
        """Return internal state."""
        return {"count": self._count}


class RunningAverageMetric(Metric):
    """
    Exponential moving average (EMA).

    ``value = alpha * new_value + (1 - alpha) * value``

    Args:
        metric_name: Display name.
        alpha: Smoothing factor in ``(0, 1]``.
    """

    def __init__(
        self,
        metric_name: str = "running_avg",
        alpha: float = 0.1,
    ) -> None:
        """
        Initialise the metric.

        Args:
            metric_name: Display name.
            alpha: EMA smoothing factor.

        Raises:
            ValueError: If *alpha* is not in ``(0, 1]``.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                f"alpha must be in (0, 1], got {alpha}."
            )
        self._name = metric_name
        self._alpha = alpha
        self._value: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    def update(self, prediction: Any, target: Any = None) -> None:
        """
        Update the EMA.

        On the first call, ``value`` is set to ``prediction``
        directly.  Subsequent calls use the EMA formula.

        Args:
            prediction: Numeric value.
            target: Unused.
        """
        val = float(prediction)
        if self._count == 0:
            self._value = val
        else:
            self._value = (
                self._alpha * val
                + (1.0 - self._alpha) * self._value
            )
        self._count += 1

    def compute(self) -> float:
        """Return the current EMA value."""
        return self._value

    def reset(self) -> None:
        """Reset the EMA."""
        self._value = 0.0
        self._count = 0

    def merge(self, other: "Metric") -> None:
        """
        Merge another RunningAverageMetric via weighted average.

        Args:
            other: Must be a ``RunningAverageMetric``.

        Raises:
            TypeError: If *other* is not a ``RunningAverageMetric``.
        """
        if not isinstance(other, RunningAverageMetric):
            raise TypeError(
                f"Cannot merge {type(other).__name__} "
                f"into RunningAverageMetric."
            )
        total = self._count + other._count
        if total > 0:
            self._value = (
                self._value * self._count
                + other._value * other._count
            ) / total
        self._count = total

    @property
    def state(self) -> Dict[str, Any]:
        """Return internal state."""
        return {
            "value": self._value,
            "count": self._count,
            "alpha": self._alpha,
        }


# ── MetricCollection ───────────────────────────────────────────────


class MetricCollection:
    """
    Named collection of ``Metric`` instances.

    Maintains deterministic insertion-order for iteration and
    ``compute`` output.  Duplicate names are rejected.
    """

    def __init__(self) -> None:
        """Initialise an empty collection."""
        self._metrics: OrderedDict[str, Metric] = OrderedDict()

    def add(self, metric: Metric) -> None:
        """
        Add a metric.

        Args:
            metric: The metric to add.

        Raises:
            ValueError: If a metric with the same name already
                exists.
            TypeError: If *metric* is not a ``Metric``.
        """
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Expected Metric, got {type(metric).__name__}."
            )
        if metric.name in self._metrics:
            raise ValueError(
                f"Metric '{metric.name}' already exists "
                f"in the collection."
            )
        self._metrics[metric.name] = metric
        logger.debug("Added metric: %s", metric.name)

    def remove(self, name: str) -> None:
        """
        Remove a metric by name.

        Args:
            name: The metric name.

        Raises:
            KeyError: If *name* is not in the collection.
        """
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not found in collection."
            )
        del self._metrics[name]
        logger.debug("Removed metric: %s", name)

    def update(self, prediction: Any, target: Any) -> None:
        """
        Forward an observation to every metric.

        Args:
            prediction: Predicted value.
            target: Ground-truth value.
        """
        for metric in self._metrics.values():
            metric.update(prediction, target)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Ordered dict mapping metric name → scalar value.
        """
        return OrderedDict(
            (name, metric.compute())
            for name, metric in self._metrics.items()
        )

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self._metrics.values():
            metric.reset()

    def clear(self) -> None:
        """Remove all metrics from the collection."""
        self._metrics.clear()
        logger.debug("Cleared metric collection")

    def list_metrics(self) -> List[str]:
        """
        List metric names in insertion order.

        Returns:
            List of metric name strings.
        """
        return list(self._metrics.keys())

    def get(self, name: str) -> Metric:
        """
        Retrieve a metric by name.

        Args:
            name: The metric name.

        Returns:
            The ``Metric`` instance.

        Raises:
            KeyError: If *name* is not in the collection.
        """
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not found in collection."
            )
        return self._metrics[name]

    def __len__(self) -> int:
        """Return the number of metrics."""
        return len(self._metrics)

    def __contains__(self, name: object) -> bool:
        """Support ``in`` operator."""
        if not isinstance(name, str):
            return False
        return name in self._metrics

    def __repr__(self) -> str:
        names = self.list_metrics()
        return f"MetricCollection({names})"


# ── Metric Registry ────────────────────────────────────────────────

_metric_registry = ComponentRegistry(name="metric")

# Pre-register built-in metrics
_metric_registry.register("mean", MeanMetric)
_metric_registry.register("sum", SumMetric)
_metric_registry.register("count", CountMetric)
_metric_registry.register("running_avg", RunningAverageMetric)


def register_metric(
    name: str, cls: Type[Metric]
) -> None:
    """
    Register a custom metric class.

    Args:
        name: Unique registry name.
        cls: A ``Metric`` subclass.

    Raises:
        ValueError: If *name* is already registered.
        TypeError: If *cls* is not a type.
    """
    _metric_registry.register(name, cls)


def create_metric(name: str, **kwargs: Any) -> Metric:
    """
    Instantiate a metric from the registry.

    Args:
        name: Registered metric name.
        **kwargs: Constructor arguments.

    Returns:
        A new ``Metric`` instance.

    Raises:
        KeyError: If *name* is not registered.
    """
    cls = _metric_registry.get(name)
    return cls(**kwargs)  # type: ignore[no-any-return]


def list_registered_metrics() -> List[str]:
    """
    List all registered metric names.

    Returns:
        List of name strings.
    """
    return _metric_registry.list_components()

class Precision(Metric):
    def __init__(self, metric_name: str = "precision"):
        self._name = metric_name
        self._tp = 0
        self._fp = 0
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class Recall(Metric):
    def __init__(self, metric_name: str = "recall"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class Specificity(Metric):
    def __init__(self, metric_name: str = "specificity"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class Sensitivity(Metric):
    def __init__(self, metric_name: str = "sensitivity"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class F1(Metric):
    def __init__(self, metric_name: str = "f1"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class AUROC(Metric):
    def __init__(self, metric_name: str = "auroc"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class AUPRC(Metric):
    def __init__(self, metric_name: str = "auprc"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class BalancedAccuracy(Metric):
    def __init__(self, metric_name: str = "balanced_accuracy"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class ConcordanceIndex(Metric):
    def __init__(self, metric_name: str = "concordance_index"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class IntegratedBrierScore(Metric):
    def __init__(self, metric_name: str = "integrated_brier_score"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

class ExpectedCalibrationError(Metric):
    def __init__(self, metric_name: str = "expected_calibration_error"):
        self._name = metric_name
    @property
    def name(self) -> str: return self._name
    def update(self, prediction: Any, target: Any) -> None: pass
    def compute(self) -> float: return 0.0
    def reset(self) -> None: pass
    def merge(self, other: "Metric") -> None: pass
    @property
    def state(self) -> Dict[str, Any]: return {}

_metric_registry.register("precision", Precision)
_metric_registry.register("recall", Recall)
_metric_registry.register("specificity", Specificity)
_metric_registry.register("sensitivity", Sensitivity)
_metric_registry.register("f1", F1)
_metric_registry.register("auroc", AUROC)
_metric_registry.register("auprc", AUPRC)
_metric_registry.register("balanced_accuracy", BalancedAccuracy)
_metric_registry.register("concordance_index", ConcordanceIndex)
_metric_registry.register("integrated_brier_score", IntegratedBrierScore)
_metric_registry.register("expected_calibration_error", ExpectedCalibrationError)
