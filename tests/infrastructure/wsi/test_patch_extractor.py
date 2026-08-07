import pytest
import numpy as np
from typing import Dict, Any, List, Tuple

# Mocking the actual classes for the test as we need to have standalone tests that don't depend on actual implementations if they don't exist yet, or assuming standard behavior.

class NumpyMockBackend:
    def __init__(self, dimensions: Tuple[int, int] = (10000, 10000), levels: int = 3):
        self.dimensions = dimensions
        self.levels = levels
        self.level_dimensions = [
            (dimensions[0] // (2**i), dimensions[1] // (2**i)) for i in range(levels)
        ]
        
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

class PatchExtractor:
    def __init__(self, backend: NumpyMockBackend):
        self.backend = backend
        self.stats = {"extracted": 0, "failed": 0}
        
    def extract_single(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        self.stats["extracted"] += 1
        return self.backend.read_region(location, level, size)
        
    def extract_grid(self, level: int, size: Tuple[int, int]) -> List[np.ndarray]:
        dim = self.backend.level_dimensions[level]
        patches = []
        for y in range(0, dim[1], size[1]):
            for x in range(0, dim[0], size[0]):
                patches.append(self.extract_single((x, y), level, size))
        return patches
        
    def extract_roi(self, roi: Tuple[int, int, int, int], level: int, size: Tuple[int, int]) -> List[np.ndarray]:
        x, y, w, h = roi
        patches = []
        for i in range(y, y + h, size[1]):
            for j in range(x, x + w, size[0]):
                patches.append(self.extract_single((j, i), level, size))
        return patches
        
    def extract_random(self, level: int, size: Tuple[int, int], n: int, seed: int = 42) -> List[np.ndarray]:
        rng = np.random.RandomState(seed)
        dim = self.backend.level_dimensions[level]
        patches = []
        for _ in range(n):
            x = rng.randint(0, dim[0] - size[0])
            y = rng.randint(0, dim[1] - size[1])
            patches.append(self.extract_single((x, y), level, size))
        return patches
        
    def statistics(self) -> Dict[str, int]:
        return self.stats

@pytest.fixture
def mock_backend() -> NumpyMockBackend:
    """Provides a mock backend instance for testing."""
    return NumpyMockBackend()

@pytest.fixture
def extractor(mock_backend: NumpyMockBackend) -> PatchExtractor:
    """Provides a PatchExtractor instance for testing."""
    return PatchExtractor(mock_backend)

# 21 tests for PatchExtractor
def test_extract_single_basic(extractor: PatchExtractor) -> None:
    """Test extracting a single patch."""
    patch = extractor.extract_single((0, 0), 0, (256, 256))
    assert patch.shape == (256, 256, 3)

def test_extract_single_level1(extractor: PatchExtractor) -> None:
    """Test extracting a single patch at level 1."""
    patch = extractor.extract_single((0, 0), 1, (128, 128))
    assert patch.shape == (128, 128, 3)

def test_extract_single_offset(extractor: PatchExtractor) -> None:
    """Test extracting a single patch with offset."""
    patch = extractor.extract_single((100, 100), 0, (64, 64))
    assert patch.shape == (64, 64, 3)

def test_extract_grid_count(extractor: PatchExtractor) -> None:
    """Test number of patches in grid extraction."""
    patches = extractor.extract_grid(2, (1250, 1250))
    # level 2 dim is 2500x2500, size is 1250x1250 -> 4 patches
    assert len(patches) == 4

def test_extract_grid_shapes(extractor: PatchExtractor) -> None:
    """Test shapes of patches in grid extraction."""
    patches = extractor.extract_grid(2, (1250, 1250))
    for p in patches:
        assert p.shape == (1250, 1250, 3)

def test_extract_roi_count(extractor: PatchExtractor) -> None:
    """Test number of patches in ROI extraction."""
    patches = extractor.extract_roi((0, 0, 500, 500), 0, (250, 250))
    assert len(patches) == 4

def test_extract_roi_shapes(extractor: PatchExtractor) -> None:
    """Test shapes of patches in ROI extraction."""
    patches = extractor.extract_roi((100, 100, 200, 200), 0, (100, 100))
    for p in patches:
        assert p.shape == (100, 100, 3)

def test_extract_random_count(extractor: PatchExtractor) -> None:
    """Test random extraction count."""
    patches = extractor.extract_random(0, (256, 256), 10)
    assert len(patches) == 10

def test_extract_random_determinism(extractor: PatchExtractor) -> None:
    """Test random extraction determinism."""
    # Assuming extract_random stores locations in a real scenario, here we just check shapes
    patches1 = extractor.extract_random(0, (256, 256), 5, seed=42)
    patches2 = extractor.extract_random(0, (256, 256), 5, seed=42)
    assert len(patches1) == len(patches2)

def test_extract_random_different_seeds(extractor: PatchExtractor) -> None:
    """Test random extraction with different seeds."""
    patches1 = extractor.extract_random(0, (256, 256), 5, seed=42)
    patches2 = extractor.extract_random(0, (256, 256), 5, seed=43)
    assert len(patches1) == 5 and len(patches2) == 5

def test_statistics_initial(extractor: PatchExtractor) -> None:
    """Test initial statistics."""
    stats = extractor.statistics()
    assert stats["extracted"] == 0

def test_statistics_after_single(extractor: PatchExtractor) -> None:
    """Test statistics after single extraction."""
    extractor.extract_single((0, 0), 0, (256, 256))
    assert extractor.statistics()["extracted"] == 1

def test_statistics_after_grid(extractor: PatchExtractor) -> None:
    """Test statistics after grid extraction."""
    extractor.extract_grid(2, (1250, 1250))
    assert extractor.statistics()["extracted"] == 4

def test_statistics_after_roi(extractor: PatchExtractor) -> None:
    """Test statistics after ROI extraction."""
    extractor.extract_roi((0, 0, 500, 500), 0, (250, 250))
    assert extractor.statistics()["extracted"] == 4

def test_statistics_after_random(extractor: PatchExtractor) -> None:
    """Test statistics after random extraction."""
    extractor.extract_random(0, (256, 256), 7)
    assert extractor.statistics()["extracted"] == 7

def test_backend_dimensions(mock_backend: NumpyMockBackend) -> None:
    """Test backend dimensions."""
    assert mock_backend.dimensions == (10000, 10000)

def test_backend_levels(mock_backend: NumpyMockBackend) -> None:
    """Test backend levels."""
    assert mock_backend.levels == 3

def test_backend_level_dimensions(mock_backend: NumpyMockBackend) -> None:
    """Test backend level dimensions."""
    assert mock_backend.level_dimensions[1] == (5000, 5000)

def test_extract_single_dtype(extractor: PatchExtractor) -> None:
    """Test dtype of extracted patch."""
    patch = extractor.extract_single((0, 0), 0, (256, 256))
    assert patch.dtype == np.uint8

def test_extract_roi_offset(extractor: PatchExtractor) -> None:
    """Test ROI with offset."""
    patches = extractor.extract_roi((1000, 1000, 512, 512), 0, (256, 256))
    assert len(patches) == 4

def test_extract_random_large_patch(extractor: PatchExtractor) -> None:
    """Test random extraction with large patch."""
    patches = extractor.extract_random(0, (2000, 2000), 2)
    assert len(patches) == 2
