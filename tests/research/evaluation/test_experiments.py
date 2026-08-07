"""Tests for experiment management."""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExperimentRun:
    """Mock experiment run."""
    run_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"

    def complete(self, results):
        """Complete the run with results."""
        self.results = results
        self.status = "completed"

    def fail(self):
        """Fail the run."""
        self.status = "failed"

class ExperimentRegistry:
    """Mock experiment registry."""
    def __init__(self):
        self.experiments = {}

    def register(self, run: ExperimentRun):
        """Register an experiment run."""
        if run.run_id in self.experiments:
            raise ValueError(f"Experiment {run.run_id} already exists")
        self.experiments[run.run_id] = run

    def get(self, run_id):
        """Get an experiment run by ID."""
        if run_id not in self.experiments:
            raise KeyError(f"Experiment {run_id} not found")
        return self.experiments[run_id]

@pytest.fixture
def registry():
    """Fixture for ExperimentRegistry."""
    return ExperimentRegistry()

@pytest.mark.parametrize("run_id", ["run1", "run2", "test-run", "12345", "exp_001"])
def test_experiment_run_init(run_id):
    """Test ExperimentRun initialization."""
    run = ExperimentRun(run_id=run_id)
    assert run.run_id == run_id
    assert run.status == "created"

@pytest.mark.parametrize("results", [{"acc": 0.9}, {"loss": 0.1, "f1": 0.8}, {}, {"a": 1, "b": 2}, {"score": 100}])
def test_experiment_run_complete(results):
    """Test ExperimentRun completion."""
    run = ExperimentRun(run_id="test")
    run.complete(results)
    assert run.status == "completed"
    assert run.results == results

@pytest.mark.parametrize("run_id", ["fail1", "fail2", "fail3", "f4", "f5"])
def test_experiment_run_fail(run_id):
    """Test ExperimentRun failure."""
    run = ExperimentRun(run_id=run_id)
    run.fail()
    assert run.status == "failed"

@pytest.mark.parametrize("run_id", ["r1", "r2", "r3", "r4", "r5"])
def test_registry_register_and_get(registry, run_id):
    """Test registering and retrieving an experiment."""
    run = ExperimentRun(run_id=run_id)
    registry.register(run)
    retrieved = registry.get(run_id)
    assert retrieved == run

@pytest.mark.parametrize("run_id", ["r1", "r2", "r3", "r4", "r5"])
def test_registry_register_duplicate(registry, run_id):
    """Test registering duplicate experiments."""
    run = ExperimentRun(run_id=run_id)
    registry.register(run)
    with pytest.raises(ValueError):
        registry.register(run)

@pytest.mark.parametrize("run_id", ["missing1", "missing2", "missing3", "m4", "m5"])
def test_registry_get_missing(registry, run_id):
    """Test retrieving missing experiments."""
    with pytest.raises(KeyError):
        registry.get(run_id)
