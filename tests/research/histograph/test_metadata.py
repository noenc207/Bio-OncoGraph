import pytest
import dataclasses
from typing import Dict, Any

from bio_oncograph.research.histograph.metadata import GraphMetadata


def test_metadata_initialization() -> None:
    """Test GraphMetadata initialization with required fields."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    assert meta.graph_id == "g1"
    assert meta.patient_id == "p1"
    assert meta.slide_id == "s1"


def test_metadata_default_fields() -> None:
    """Test GraphMetadata default fields."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    assert meta.magnification is None
    assert meta.stain_type is None
    assert isinstance(meta.extra_info, dict)
    assert len(meta.extra_info) == 0


def test_metadata_custom_fields() -> None:
    """Test GraphMetadata initialization with custom fields."""
    meta = GraphMetadata(
        graph_id="g1",
        patient_id="p1",
        slide_id="s1",
        magnification=20.0,
        stain_type="H&E",
        extra_info={"scanner": "Aperio"}
    )
    assert meta.magnification == 20.0
    assert meta.stain_type == "H&E"
    assert meta.extra_info["scanner"] == "Aperio"


def test_metadata_is_frozen() -> None:
    """Test that GraphMetadata is frozen."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.graph_id = "g2"


def test_metadata_equality() -> None:
    """Test equality operator for GraphMetadata."""
    meta1 = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    meta2 = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    assert meta1 == meta2


def test_metadata_inequality() -> None:
    """Test inequality operator for GraphMetadata."""
    meta1 = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    meta2 = GraphMetadata(graph_id="g2", patient_id="p1", slide_id="s1")
    assert meta1 != meta2


def test_metadata_to_dict() -> None:
    """Test serialization to dictionary."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    d = meta.to_dict()
    assert d["graph_id"] == "g1"
    assert d["patient_id"] == "p1"


def test_metadata_from_dict() -> None:
    """Test deserialization from dictionary."""
    d = {
        "graph_id": "g3",
        "patient_id": "p3",
        "slide_id": "s3",
        "magnification": 40.0
    }
    meta = GraphMetadata.from_dict(d)
    assert meta.graph_id == "g3"
    assert meta.magnification == 40.0


def test_metadata_hashable() -> None:
    """Test that GraphMetadata is hashable."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1")
    d = {meta: "data"}
    assert d[meta] == "data"


def test_metadata_missing_required() -> None:
    """Test that missing required fields raises error."""
    with pytest.raises(TypeError):
        GraphMetadata(patient_id="p1", slide_id="s1")


def test_metadata_extra_info_immutability_warning() -> None:
    """Test that extra_info dict is mutable but class is frozen (standard dataclass behavior)."""
    meta = GraphMetadata(graph_id="g1", patient_id="p1", slide_id="s1", extra_info={"k": "v"})
    meta.extra_info["k"] = "v2"
    assert meta.extra_info["k"] == "v2"


def test_metadata_from_dict_missing_fields() -> None:
    """Test from_dict raises error if missing fields."""
    with pytest.raises(TypeError):
        GraphMetadata.from_dict({"graph_id": "g1"})
