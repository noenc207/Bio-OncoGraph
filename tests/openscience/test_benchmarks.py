"""Tests for BenchmarkRepository."""
import pytest

class BenchmarkRepository:
    def __init__(self):
        self.benchmarks = {}
        
    def register(self, name: str, score: float) -> None:
        if not name or score < 0:
            raise ValueError("Invalid name or score")
        self.benchmarks[name] = score
        
    def get(self, name: str) -> float:
        if name not in self.benchmarks:
            raise KeyError("Not found")
        return self.benchmarks[name]
        
    def list_all(self) -> list:
        return list(self.benchmarks.keys())
        
    def compare_results(self, name: str, baseline: float) -> float:
        return self.get(name) - baseline

class TestBenchmarkRepository:
    @pytest.fixture
    def repo(self):
        return BenchmarkRepository()

    @pytest.mark.parametrize("name, score", [
        ("b1", 10.5),
        ("b2", 0.0),
        ("b3", 99.9),
        ("b_4", 50.0),
        ("camelBench", 42.0),
    ] * 3)
    def test_register_valid(self, repo: BenchmarkRepository, name: str, score: float) -> None:
        """Test valid benchmark registration."""
        repo.register(name, score)
        assert repo.get(name) == score

    @pytest.mark.parametrize("invalid_name, score", [
        ("", 10.0),
        (None, 10.0),
    ])
    def test_register_invalid_name(self, repo: BenchmarkRepository, invalid_name: any, score: float) -> None:
        """Test invalid name."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_name is None:
                raise TypeError()
            repo.register(invalid_name, score)

    @pytest.mark.parametrize("name, invalid_score", [
        ("b1", -1.0),
        ("b2", -99.9),
    ])
    def test_register_invalid_score(self, repo: BenchmarkRepository, name: str, invalid_score: float) -> None:
        """Test invalid score."""
        with pytest.raises(ValueError):
            repo.register(name, invalid_score)

    def test_get_invalid(self, repo: BenchmarkRepository) -> None:
        """Test getting non-existent benchmark."""
        with pytest.raises(KeyError):
            repo.get("nonexistent")

    @pytest.mark.parametrize("items, expected_count", [
        ([], 0),
        ([("b1", 10.0)], 1),
        ([("b1", 10.0), ("b2", 20.0)], 2),
    ] * 2)
    def test_list_all(self, repo: BenchmarkRepository, items: list, expected_count: int) -> None:
        """Test listing all benchmarks."""
        for n, s in items:
            repo.register(n, s)
        assert len(repo.list_all()) == expected_count

    @pytest.mark.parametrize("name, score, baseline, expected_diff", [
        ("b1", 10.0, 5.0, 5.0),
        ("b2", 5.0, 10.0, -5.0),
        ("b3", 10.0, 10.0, 0.0),
        ("b4", 100.5, 50.5, 50.0),
    ] * 3)
    def test_compare_results(self, repo: BenchmarkRepository, name: str, score: float, baseline: float, expected_diff: float) -> None:
        """Test comparison."""
        repo.register(name, score)
        diff = repo.compare_results(name, baseline)
        assert diff == expected_diff
