"""
Tests for SlideMetadata and PatchMetadata dataclasses.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class SlideMetadata:
    slide_id: str
    vendor: str
    mpp_x: float
    mpp_y: float
    objective_power: int
    properties: Dict[str, Any]

@dataclass(frozen=True)
class PatchMetadata:
    patch_id: str
    slide_id: str
    x: int
    y: int
    level: int
    size: int
    tissue_percentage: float

def test_slide_metadata_creation() -> None:
    """Test creating SlideMetadata."""
    meta = SlideMetadata(
        slide_id="slide_001",
        vendor="aperio",
        mpp_x=0.25,
        mpp_y=0.25,
        objective_power=40,
        properties={"key": "value"}
    )
    assert meta.slide_id == "slide_001"
    assert meta.vendor == "aperio"
    assert meta.mpp_x == 0.25
    assert meta.mpp_y == 0.25
    assert meta.objective_power == 40
    assert meta.properties["key"] == "value"

def test_slide_metadata_frozen() -> None:
    """Test SlideMetadata immutability."""
    meta = SlideMetadata("s1", "v", 0.5, 0.5, 20, {})
    with pytest.raises(Exception):
        meta.slide_id = "s2"  # type: ignore

def test_patch_metadata_creation() -> None:
    """Test creating PatchMetadata."""
    meta = PatchMetadata(
        patch_id="p_001",
        slide_id="s_001",
        x=100,
        y=200,
        level=0,
        size=256,
        tissue_percentage=0.95
    )
    assert meta.patch_id == "p_001"
    assert meta.slide_id == "s_001"
    assert meta.x == 100
    assert meta.y == 200
    assert meta.level == 0
    assert meta.size == 256
    assert meta.tissue_percentage == 0.95

def test_patch_metadata_frozen() -> None:
    """Test PatchMetadata immutability."""
    meta = PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0)
    with pytest.raises(Exception):
        meta.x = 10  # type: ignore

def test_slide_metadata_equality() -> None:
    """Test SlideMetadata equality."""
    meta1 = SlideMetadata("s1", "v", 0.5, 0.5, 20, {"k": "v"})
    meta2 = SlideMetadata("s1", "v", 0.5, 0.5, 20, {"k": "v"})
    assert meta1 == meta2

def test_slide_metadata_inequality() -> None:
    """Test SlideMetadata inequality."""
    meta1 = SlideMetadata("s1", "v", 0.5, 0.5, 20, {})
    meta2 = SlideMetadata("s2", "v", 0.5, 0.5, 20, {})
    assert meta1 != meta2

def test_patch_metadata_equality() -> None:
    """Test PatchMetadata equality."""
    meta1 = PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0)
    meta2 = PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0)
    assert meta1 == meta2

def test_patch_metadata_inequality() -> None:
    """Test PatchMetadata inequality."""
    meta1 = PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0)
    meta2 = PatchMetadata("p2", "s1", 0, 0, 0, 256, 1.0)
    assert meta1 != meta2

def test_slide_metadata_hashable() -> None:
    """Test SlideMetadata can be hashed (if dictionary is avoided)."""
    # Dataclasses with dict are not hashable by default, but we can verify it fails as expected or test properties.
    meta = SlideMetadata("s1", "v", 0.5, 0.5, 20, {})
    with pytest.raises(TypeError):
        hash(meta)

def test_patch_metadata_hashable() -> None:
    """Test PatchMetadata is hashable."""
    meta = PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0)
    assert hash(meta) == hash(PatchMetadata("p1", "s1", 0, 0, 0, 256, 1.0))

def test_patch_metadata_zero_tissue() -> None:
    """Test PatchMetadata with zero tissue."""
    meta = PatchMetadata("p1", "s1", 10, 10, 1, 256, 0.0)
    assert meta.tissue_percentage == 0.0

def test_slide_metadata_empty_properties() -> None:
    """Test SlideMetadata with empty properties dict."""
    meta = SlideMetadata("s1", "v", 1.0, 1.0, 10, {})
    assert len(meta.properties) == 0

def test_patch_metadata_negative_coordinates() -> None:
    """Test PatchMetadata allows negative coordinates if needed."""
    meta = PatchMetadata("p1", "s1", -10, -20, 0, 256, 0.5)
    assert meta.x == -10
    assert meta.y == -20
