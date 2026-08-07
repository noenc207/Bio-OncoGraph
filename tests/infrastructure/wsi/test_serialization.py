import pytest
import json
from typing import Dict, Any

class PatchSerializer:
    @staticmethod
    def serialize(patch_data: Dict[str, Any]) -> str:
        return json.dumps(patch_data)
        
    @staticmethod
    def deserialize(data_str: str) -> Dict[str, Any]:
        return json.loads(data_str)

class SlideIndexSerializer:
    @staticmethod
    def serialize(index_data: Dict[str, Any], version: str = "1.0") -> str:
        return json.dumps({"version": version, "data": index_data})
        
    @staticmethod
    def deserialize(data_str: str) -> Dict[str, Any]:
        return json.loads(data_str)

@pytest.fixture
def sample_patch() -> Dict[str, Any]:
    """Sample patch dictionary."""
    return {"x": 100, "y": 200, "label": "normal"}

@pytest.fixture
def sample_index() -> Dict[str, Any]:
    """Sample index dictionary."""
    return {"slide_id": "s1", "patches": ["p1", "p2"]}

# 13 tests
def test_patch_serialize(sample_patch: Dict[str, Any]) -> None:
    """Test patch serialization."""
    s = PatchSerializer.serialize(sample_patch)
    assert isinstance(s, str)
    assert "normal" in s

def test_patch_deserialize(sample_patch: Dict[str, Any]) -> None:
    """Test patch deserialization."""
    s = PatchSerializer.serialize(sample_patch)
    d = PatchSerializer.deserialize(s)
    assert d == sample_patch

def test_patch_roundtrip() -> None:
    """Test patch roundtrip."""
    data = {"a": 1, "b": [1, 2, 3]}
    assert PatchSerializer.deserialize(PatchSerializer.serialize(data)) == data

def test_patch_deserialize_invalid() -> None:
    """Test patch deserialize invalid JSON."""
    with pytest.raises(json.JSONDecodeError):
        PatchSerializer.deserialize("{invalid}")

def test_index_serialize(sample_index: Dict[str, Any]) -> None:
    """Test index serialization."""
    s = SlideIndexSerializer.serialize(sample_index)
    assert "version" in s
    assert "s1" in s

def test_index_serialize_custom_version(sample_index: Dict[str, Any]) -> None:
    """Test index serialization with custom version."""
    s = SlideIndexSerializer.serialize(sample_index, version="2.0")
    d = json.loads(s)
    assert d["version"] == "2.0"

def test_index_deserialize(sample_index: Dict[str, Any]) -> None:
    """Test index deserialization."""
    s = SlideIndexSerializer.serialize(sample_index)
    d = SlideIndexSerializer.deserialize(s)
    assert d["data"] == sample_index
    assert d["version"] == "1.0"

def test_index_roundtrip(sample_index: Dict[str, Any]) -> None:
    """Test index roundtrip."""
    s = SlideIndexSerializer.serialize(sample_index)
    d = SlideIndexSerializer.deserialize(s)
    assert d["data"] == sample_index

def test_index_deserialize_invalid() -> None:
    """Test index deserialize invalid."""
    with pytest.raises(json.JSONDecodeError):
        SlideIndexSerializer.deserialize("not json")

def test_patch_empty_serialize() -> None:
    """Test serialize empty dict."""
    assert PatchSerializer.serialize({}) == "{}"

def test_index_empty_serialize() -> None:
    """Test serialize empty index."""
    s = SlideIndexSerializer.serialize({})
    d = json.loads(s)
    assert d["data"] == {}

def test_patch_nested_serialize() -> None:
    """Test patch serialize nested data."""
    data = {"meta": {"res": 0.5}, "tags": ["a", "b"]}
    s = PatchSerializer.serialize(data)
    assert "meta" in s

def test_index_nested_serialize() -> None:
    """Test index serialize nested data."""
    data = {"slides": [{"id": "s1"}, {"id": "s2"}]}
    s = SlideIndexSerializer.serialize(data)
    d = SlideIndexSerializer.deserialize(s)
    assert len(d["data"]["slides"]) == 2
