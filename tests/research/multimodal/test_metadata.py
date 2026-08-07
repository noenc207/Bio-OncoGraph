import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List

# ==============================================================================
# MOCKS
# ==============================================================================
@dataclass(frozen=True)
class ModalityMetadata:
    name: str
    dimension: int
    data_type: str
    description: str = ""

@dataclass(frozen=True)
class FusionMetadata:
    strategy_name: str
    input_modalities: List[str]
    output_dimension: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

# ==============================================================================
# TESTS
# ==============================================================================

def test_modality_metadata_creation() -> None:
    """Test creating ModalityMetadata."""
    meta = ModalityMetadata(name="clinical", dimension=128, data_type="tabular", description="Patient records")
    assert meta.name == "clinical"
    assert meta.dimension == 128
    assert meta.data_type == "tabular"
    assert meta.description == "Patient records"

def test_modality_metadata_immutability() -> None:
    """Test ModalityMetadata is immutable."""
    meta = ModalityMetadata(name="genomics", dimension=512, data_type="sequence")
    with pytest.raises(Exception):
        meta.dimension = 256 # type: ignore

def test_fusion_metadata_creation() -> None:
    """Test creating FusionMetadata."""
    meta = FusionMetadata(
        strategy_name="attention",
        input_modalities=["clinical", "genomics"],
        output_dimension=256,
        hyperparameters={"heads": 4}
    )
    assert meta.strategy_name == "attention"
    assert meta.input_modalities == ["clinical", "genomics"]
    assert meta.output_dimension == 256
    assert meta.hyperparameters == {"heads": 4}

def test_fusion_metadata_immutability() -> None:
    """Test FusionMetadata is immutable."""
    meta = FusionMetadata(strategy_name="early", input_modalities=["a", "b"], output_dimension=10)
    with pytest.raises(Exception):
        meta.strategy_name = "late" # type: ignore

@pytest.mark.parametrize("dim", [10, 100, 1024])
def test_modality_metadata_dimensions(dim: int) -> None:
    """Test ModalityMetadata with various dimensions."""
    meta = ModalityMetadata(name="img", dimension=dim, data_type="image")
    assert meta.dimension == dim

@pytest.mark.parametrize("mods", [
    [],
    ["single"],
    ["clin", "gen", "path", "rad"]
])
def test_fusion_metadata_modalities(mods: List[str]) -> None:
    """Test FusionMetadata with varying modalities."""
    meta = FusionMetadata(strategy_name="concat", input_modalities=mods, output_dimension=100)
    assert meta.input_modalities == mods
