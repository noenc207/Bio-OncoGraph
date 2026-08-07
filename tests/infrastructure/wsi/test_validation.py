import pytest
from typing import Tuple, List

class SlideValidator:
    def __init__(self, dimensions: Tuple[int, int], levels: int, supported_formats: List[str]):
        self.dimensions = dimensions
        self.levels = levels
        self.supported_formats = supported_formats

    def validate_coords(self, coords: Tuple[int, int]) -> bool:
        return 0 <= coords[0] < self.dimensions[0] and 0 <= coords[1] < self.dimensions[1]

    def validate_patch(self, coords: Tuple[int, int], size: Tuple[int, int]) -> bool:
        return (self.validate_coords(coords) and 
                coords[0] + size[0] <= self.dimensions[0] and 
                coords[1] + size[1] <= self.dimensions[1])

    def validate_level(self, level: int) -> bool:
        return 0 <= level < self.levels

    def validate_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = roi
        return self.validate_patch((x, y), (w, h))

    def validate_format(self, fmt: str) -> bool:
        return fmt.lower() in self.supported_formats

@pytest.fixture
def validator() -> SlideValidator:
    """Provides a SlideValidator instance."""
    return SlideValidator((10000, 10000), 5, ["tiff", "svs", "ndpi"])

# 16 tests
def test_valid_coords_origin(validator: SlideValidator) -> None:
    """Test origin coords."""
    assert validator.validate_coords((0, 0)) is True

def test_valid_coords_inside(validator: SlideValidator) -> None:
    """Test inside coords."""
    assert validator.validate_coords((5000, 5000)) is True

def test_invalid_coords_negative(validator: SlideValidator) -> None:
    """Test negative coords."""
    assert validator.validate_coords((-1, 0)) is False

def test_invalid_coords_out_of_bounds(validator: SlideValidator) -> None:
    """Test out of bounds coords."""
    assert validator.validate_coords((10001, 5000)) is False

def test_valid_patch_inside(validator: SlideValidator) -> None:
    """Test valid patch."""
    assert validator.validate_patch((0, 0), (256, 256)) is True

def test_invalid_patch_overflow_x(validator: SlideValidator) -> None:
    """Test patch overflow x."""
    assert validator.validate_patch((9800, 0), (256, 256)) is False

def test_invalid_patch_overflow_y(validator: SlideValidator) -> None:
    """Test patch overflow y."""
    assert validator.validate_patch((0, 9900), (256, 256)) is False

def test_valid_level(validator: SlideValidator) -> None:
    """Test valid level."""
    assert validator.validate_level(0) is True
    assert validator.validate_level(4) is True

def test_invalid_level_negative(validator: SlideValidator) -> None:
    """Test negative level."""
    assert validator.validate_level(-1) is False

def test_invalid_level_too_high(validator: SlideValidator) -> None:
    """Test level too high."""
    assert validator.validate_level(5) is False

def test_valid_roi(validator: SlideValidator) -> None:
    """Test valid ROI."""
    assert validator.validate_roi((1000, 1000, 5000, 5000)) is True

def test_invalid_roi_overflow(validator: SlideValidator) -> None:
    """Test ROI overflow."""
    assert validator.validate_roi((5000, 5000, 6000, 6000)) is False

def test_invalid_roi_negative(validator: SlideValidator) -> None:
    """Test negative ROI."""
    assert validator.validate_roi((-100, 0, 500, 500)) is False

def test_valid_format_svs(validator: SlideValidator) -> None:
    """Test valid format svs."""
    assert validator.validate_format("svs") is True

def test_valid_format_case_insensitive(validator: SlideValidator) -> None:
    """Test valid format uppercase."""
    assert validator.validate_format("TIFF") is True

def test_invalid_format(validator: SlideValidator) -> None:
    """Test invalid format."""
    assert validator.validate_format("jpeg") is False
