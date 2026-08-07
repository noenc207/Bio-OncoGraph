"""Tests for BenchmarkRunner and BenchmarkConfig."""

import pytest
import numpy as np
from typing import Dict, Any


class BenchmarkConfig:
    """Mock BenchmarkConfig."""
    def __init__(self, iterations: int = 100, warmup: int = 10) -> None:
        self.iterations = iterations
        self.warmup = warmup


class BenchmarkRunner:
    """Mock BenchmarkRunner."""
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.latencies: Dict[str, float] = {}
        self.memory_usage: float = 0.0

    def run_benchmark(self, name: str, data: np.ndarray) -> None:
        """Simulate running benchmark."""
        if self.config.iterations <= 0:
            raise ValueError("Iterations must be positive")
        latency = data.size * 0.001
        self.latencies[name] = latency
        self.memory_usage += data.nbytes / 1024.0 / 1024.0

    def get_throughput(self, name: str, items: int) -> float:
        """Calculate throughput."""
        latency = self.latencies.get(name)
        if latency is None or latency == 0:
            return 0.0
        return items / latency

    def get_memory_usage(self) -> float:
        """Get tracked memory usage in MB."""
        return self.memory_usage


@pytest.fixture
def config() -> BenchmarkConfig:
    return BenchmarkConfig()


@pytest.fixture
def runner(config: BenchmarkConfig) -> BenchmarkRunner:
    return BenchmarkRunner(config)


def test_config_init() -> None:
    cfg = BenchmarkConfig(iterations=50, warmup=5)
    assert cfg.iterations == 50
    assert cfg.warmup == 5


def test_runner_init(runner: BenchmarkRunner, config: BenchmarkConfig) -> None:
    assert runner.config == config
    assert runner.latencies == {}
    assert runner.memory_usage == 0.0


def test_run_benchmark_latency(runner: BenchmarkRunner) -> None:
    data = np.ones(1000)
    runner.run_benchmark("test1", data)
    assert "test1" in runner.latencies
    assert runner.latencies["test1"] == 1.0


def test_run_benchmark_memory(runner: BenchmarkRunner) -> None:
    data = np.ones(1024 * 1024, dtype=np.uint8)  # 1 MB
    runner.run_benchmark("test_mem", data)
    assert pytest.approx(runner.get_memory_usage(), 0.1) == 1.0


def test_run_benchmark_multiple(runner: BenchmarkRunner) -> None:
    data1 = np.ones(1000)
    data2 = np.ones(2000)
    runner.run_benchmark("test1", data1)
    runner.run_benchmark("test2", data2)
    assert runner.latencies["test1"] == 1.0
    assert runner.latencies["test2"] == 2.0


def test_run_benchmark_zero_iterations() -> None:
    cfg = BenchmarkConfig(iterations=0)
    runner = BenchmarkRunner(cfg)
    with pytest.raises(ValueError):
        runner.run_benchmark("test", np.ones(10))


def test_get_throughput_valid(runner: BenchmarkRunner) -> None:
    runner.latencies["test"] = 2.0
    assert runner.get_throughput("test", 1000) == 500.0


def test_get_throughput_missing(runner: BenchmarkRunner) -> None:
    assert runner.get_throughput("missing", 1000) == 0.0


def test_get_throughput_zero_latency(runner: BenchmarkRunner) -> None:
    runner.latencies["test"] = 0.0
    assert runner.get_throughput("test", 1000) == 0.0


def test_get_memory_usage_initial(runner: BenchmarkRunner) -> None:
    assert runner.get_memory_usage() == 0.0


def test_get_memory_usage_accumulation(runner: BenchmarkRunner) -> None:
    data = np.ones(1024 * 1024, dtype=np.uint8)
    runner.run_benchmark("test1", data)
    runner.run_benchmark("test2", data)
    assert pytest.approx(runner.get_memory_usage(), 0.1) == 2.0


def test_throughput_zero_items(runner: BenchmarkRunner) -> None:
    runner.latencies["test"] = 2.0
    assert runner.get_throughput("test", 0) == 0.0


def test_run_benchmark_empty_data(runner: BenchmarkRunner) -> None:
    data = np.array([])
    runner.run_benchmark("test_empty", data)
    assert runner.latencies["test_empty"] == 0.0


def test_config_default_values() -> None:
    cfg = BenchmarkConfig()
    assert cfg.iterations == 100
    assert cfg.warmup == 10


def test_runner_negative_latency_handling(runner: BenchmarkRunner) -> None:
    # Simulating a case where latency is manually set to negative
    runner.latencies["test_neg"] = -1.0
    assert runner.get_throughput("test_neg", 100) == -100.0


def test_runner_throughput_large_values(runner: BenchmarkRunner) -> None:
    runner.latencies["test_large"] = 0.001
    assert runner.get_throughput("test_large", 1000000) == 1000000000.0
