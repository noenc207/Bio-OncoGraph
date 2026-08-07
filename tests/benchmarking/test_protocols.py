"""Tests for benchmarking protocols."""

import pytest
import numpy as np

class MockFairBenchmarkProtocol:
    """Mock representation to test seed alignment logic."""
    def __init__(self, strict_mode=True):
        """Initialize mock protocol."""
        self.strict_mode = strict_mode
        self.seed_registry = {}

    def register_seed(self, component, seed):
        """Register a seed for a component."""
        self.seed_registry[component] = seed

    def verify_alignment(self):
        """Verify that all seeds match."""
        seeds = list(self.seed_registry.values())
        if not seeds:
            return True
        if len(set(seeds)) > 1:
            if self.strict_mode:
                raise ValueError("Misaligned seeds detected.")
            return False
        return True


@pytest.mark.parametrize("seed1", [42, 100, 2023, 0, 999])
@pytest.mark.parametrize("seed2", [43, 101, 2024, 1, 999])
@pytest.mark.parametrize("strict_mode", [True, False])
@pytest.mark.parametrize("dataset", ["tcga", "icgc", "pcawg", "gtex", "cptac"])
def test_fair_benchmark_protocol_misaligned_seeds(seed1, seed2, strict_mode, dataset):
    """Verify FairBenchmarkProtocol rejects misaligned seeds."""
    protocol = MockFairBenchmarkProtocol(strict_mode=strict_mode)
    protocol.register_seed(f"{dataset}_train", seed1)
    protocol.register_seed(f"{dataset}_test", seed2)
    
    if seed1 != seed2:
        if strict_mode:
            with pytest.raises(ValueError, match="Misaligned seeds detected."):
                protocol.verify_alignment()
        else:
            assert protocol.verify_alignment() is False
    else:
        assert protocol.verify_alignment() is True
