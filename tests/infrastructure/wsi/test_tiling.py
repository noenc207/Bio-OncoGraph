import pytest
from typing import List, Tuple
import numpy as np

class GridTiling:
    def __init__(self, slide_size: Tuple[int, int], tile_size: Tuple[int, int]):
        self.slide_size = slide_size
        self.tile_size = tile_size
        
    def get_tiles(self) -> List[Tuple[int, int]]:
        tiles = []
        for y in range(0, self.slide_size[1], self.tile_size[1]):
            for x in range(0, self.slide_size[0], self.tile_size[0]):
                tiles.append((x, y))
        return tiles

class OverlappingTiling:
    def __init__(self, slide_size: Tuple[int, int], tile_size: Tuple[int, int], overlap: Tuple[int, int]):
        self.slide_size = slide_size
        self.tile_size = tile_size
        self.overlap = overlap
        
    def get_tiles(self) -> List[Tuple[int, int]]:
        tiles = []
        stride_x = self.tile_size[0] - self.overlap[0]
        stride_y = self.tile_size[1] - self.overlap[1]
        for y in range(0, self.slide_size[1], stride_y):
            for x in range(0, self.slide_size[0], stride_x):
                tiles.append((x, y))
        return tiles

class RandomTiling:
    def __init__(self, slide_size: Tuple[int, int], tile_size: Tuple[int, int], n: int, seed: int = 42):
        self.slide_size = slide_size
        self.tile_size = tile_size
        self.n = n
        self.seed = seed
        
    def get_tiles(self) -> List[Tuple[int, int]]:
        rng = np.random.RandomState(self.seed)
        tiles = []
        for _ in range(self.n):
            x = rng.randint(0, self.slide_size[0] - self.tile_size[0])
            y = rng.randint(0, self.slide_size[1] - self.tile_size[1])
            tiles.append((x, y))
        return tiles

# 16 tests for tiling
def test_grid_tiling_count() -> None:
    """Test grid tiling tile count."""
    tiling = GridTiling((1000, 1000), (250, 250))
    assert len(tiling.get_tiles()) == 16

def test_grid_tiling_coords() -> None:
    """Test grid tiling coordinates."""
    tiling = GridTiling((500, 500), (250, 250))
    tiles = tiling.get_tiles()
    assert (0, 0) in tiles
    assert (250, 250) in tiles

def test_grid_tiling_non_divisible() -> None:
    """Test grid tiling with non-divisible sizes."""
    tiling = GridTiling((500, 500), (300, 300))
    tiles = tiling.get_tiles()
    assert len(tiles) == 4

def test_grid_tiling_coverage() -> None:
    """Test grid tiling covers origin."""
    tiling = GridTiling((1000, 1000), (500, 500))
    tiles = tiling.get_tiles()
    assert tiles[0] == (0, 0)

def test_overlapping_tiling_count() -> None:
    """Test overlapping tiling count."""
    tiling = OverlappingTiling((1000, 1000), (500, 500), (250, 250))
    # strides: 250. y: 0, 250, 500, 750 (4) x: 0, 250, 500, 750 (4) -> 16
    assert len(tiling.get_tiles()) == 16

def test_overlapping_tiling_coords() -> None:
    """Test overlapping tiling coordinates."""
    tiling = OverlappingTiling((1000, 1000), (500, 500), (250, 250))
    tiles = tiling.get_tiles()
    assert (250, 0) in tiles
    assert (250, 250) in tiles

def test_overlapping_tiling_zero_overlap() -> None:
    """Test overlapping tiling with zero overlap."""
    tiling = OverlappingTiling((1000, 1000), (500, 500), (0, 0))
    assert len(tiling.get_tiles()) == 4

def test_overlapping_tiling_large_overlap() -> None:
    """Test overlapping tiling with large overlap."""
    tiling = OverlappingTiling((500, 500), (250, 250), (200, 200))
    # stride 50. 0, 50, 100, 150, 200, 250, 300, 350, 400, 450 -> 10 x 10 = 100
    assert len(tiling.get_tiles()) == 100

def test_random_tiling_count() -> None:
    """Test random tiling count."""
    tiling = RandomTiling((1000, 1000), (250, 250), 10)
    assert len(tiling.get_tiles()) == 10

def test_random_tiling_determinism() -> None:
    """Test random tiling determinism."""
    tiling1 = RandomTiling((1000, 1000), (250, 250), 10, seed=42)
    tiling2 = RandomTiling((1000, 1000), (250, 250), 10, seed=42)
    assert tiling1.get_tiles() == tiling2.get_tiles()

def test_random_tiling_different_seeds() -> None:
    """Test random tiling different seeds."""
    tiling1 = RandomTiling((1000, 1000), (250, 250), 10, seed=42)
    tiling2 = RandomTiling((1000, 1000), (250, 250), 10, seed=43)
    assert tiling1.get_tiles() != tiling2.get_tiles()

def test_random_tiling_bounds() -> None:
    """Test random tiling bounds."""
    tiling = RandomTiling((1000, 1000), (250, 250), 100, seed=42)
    tiles = tiling.get_tiles()
    for x, y in tiles:
        assert 0 <= x <= 750
        assert 0 <= y <= 750

def test_grid_tiling_single_tile() -> None:
    """Test grid tiling with size equal to slide size."""
    tiling = GridTiling((1000, 1000), (1000, 1000))
    assert len(tiling.get_tiles()) == 1
    assert tiling.get_tiles()[0] == (0, 0)

def test_overlapping_tiling_single_tile() -> None:
    """Test overlapping tiling with size equal to slide size."""
    tiling = OverlappingTiling((1000, 1000), (1000, 1000), (500, 500))
    assert len(tiling.get_tiles()) == 1

def test_grid_tiling_larger_than_slide() -> None:
    """Test grid tiling with tile size larger than slide."""
    tiling = GridTiling((500, 500), (1000, 1000))
    assert len(tiling.get_tiles()) == 1

def test_random_tiling_zero_n() -> None:
    """Test random tiling with zero n."""
    tiling = RandomTiling((1000, 1000), (250, 250), 0)
    assert len(tiling.get_tiles()) == 0
