import pytest
from typing import Dict, Any, List

class PatchIndex:
    def __init__(self):
        self._index: Dict[str, Dict[str, Any]] = {}
        
    def add(self, patch_id: str, metadata: Dict[str, Any]) -> None:
        self._index[patch_id] = metadata
        
    def lookup(self, patch_id: str) -> Dict[str, Any]:
        return self._index.get(patch_id)
        
    def list_patches(self) -> List[str]:
        return list(self._index.keys())
        
    def size(self) -> int:
        return len(self._index)
        
    def to_dict(self) -> Dict[str, Any]:
        return self._index.copy()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatchIndex':
        idx = cls()
        idx._index = data.copy()
        return idx

@pytest.fixture
def empty_index() -> PatchIndex:
    """Provide empty PatchIndex."""
    return PatchIndex()

@pytest.fixture
def populated_index() -> PatchIndex:
    """Provide populated PatchIndex."""
    idx = PatchIndex()
    idx.add("p1", {"x": 0, "y": 0})
    idx.add("p2", {"x": 256, "y": 0})
    return idx

# 13 tests
def test_add_patch(empty_index: PatchIndex) -> None:
    """Test adding a patch to index."""
    empty_index.add("p1", {"label": "tumor"})
    assert empty_index.size() == 1

def test_lookup_existing(populated_index: PatchIndex) -> None:
    """Test looking up an existing patch."""
    assert populated_index.lookup("p1") == {"x": 0, "y": 0}

def test_lookup_missing(populated_index: PatchIndex) -> None:
    """Test looking up a missing patch."""
    assert populated_index.lookup("p3") is None

def test_list_patches(populated_index: PatchIndex) -> None:
    """Test listing patches."""
    patches = populated_index.list_patches()
    assert set(patches) == {"p1", "p2"}

def test_size_empty(empty_index: PatchIndex) -> None:
    """Test size of empty index."""
    assert empty_index.size() == 0

def test_size_populated(populated_index: PatchIndex) -> None:
    """Test size of populated index."""
    assert populated_index.size() == 2

def test_to_dict(populated_index: PatchIndex) -> None:
    """Test to_dict serialization."""
    d = populated_index.to_dict()
    assert "p1" in d and "p2" in d
    assert d["p1"]["x"] == 0

def test_from_dict_roundtrip(populated_index: PatchIndex) -> None:
    """Test roundtrip serialization."""
    d = populated_index.to_dict()
    idx = PatchIndex.from_dict(d)
    assert idx.size() == 2
    assert idx.lookup("p2") == {"x": 256, "y": 0}

def test_add_overwrite(populated_index: PatchIndex) -> None:
    """Test overwriting a patch metadata."""
    populated_index.add("p1", {"x": 100})
    assert populated_index.lookup("p1") == {"x": 100}
    assert populated_index.size() == 2

def test_add_multiple(empty_index: PatchIndex) -> None:
    """Test adding multiple patches."""
    for i in range(10):
        empty_index.add(f"p{i}", {"val": i})
    assert empty_index.size() == 10

def test_list_patches_empty(empty_index: PatchIndex) -> None:
    """Test listing patches on empty index."""
    assert empty_index.list_patches() == []

def test_from_dict_empty() -> None:
    """Test from_dict with empty dict."""
    idx = PatchIndex.from_dict({})
    assert idx.size() == 0

def test_to_dict_empty(empty_index: PatchIndex) -> None:
    """Test to_dict on empty index."""
    assert empty_index.to_dict() == {}
