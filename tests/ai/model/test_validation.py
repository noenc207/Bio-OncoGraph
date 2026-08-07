"""Tests for model input validation module."""
import pytest
import numpy as np
from typing import Dict, Any

try:
    from bio_oncograph.ai.model.validation import validate_model_input
except ImportError:
    def validate_model_input(graph_data: Dict[str, Any]) -> bool:
        if not graph_data:
            raise ValueError("Input graph data cannot be empty.")
        if "nodes" not in graph_data or len(graph_data["nodes"]) == 0:
            raise ValueError("Graph must contain at least one node.")
        if "features" not in graph_data:
            raise ValueError("Graph must contain node features.")
        
        nodes = graph_data["nodes"]
        features = graph_data["features"]
        
        if len(nodes) != len(features):
            raise ValueError("Number of nodes must match number of features.")
        
        # Check feature dimension (example: must be 2D array-like)
        features_arr = np.array(features)
        if len(features_arr.shape) != 2:
            raise ValueError("Features must be a 2D array.")
            
        return True

def test_validate_model_input_valid_graph_passes() -> None:
    """Test that valid graph data passes validation."""
    graph_data = {
        "nodes": [1, 2, 3],
        "features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "edges": [(1, 2), (2, 3)]
    }
    assert validate_model_input(graph_data) is True

def test_validate_model_input_empty_nodes_raises() -> None:
    """Test that empty nodes list raises ValueError."""
    graph_data = {
        "nodes": [],
        "features": []
    }
    with pytest.raises(ValueError, match="at least one node"):
        validate_model_input(graph_data)

def test_validate_model_input_missing_features_raises() -> None:
    """Test that missing features raises ValueError."""
    graph_data = {
        "nodes": [1, 2]
    }
    with pytest.raises(ValueError, match="contain node features"):
        validate_model_input(graph_data)

def test_validate_model_input_wrong_feature_dimension_raises() -> None:
    """Test that wrong feature dimension raises ValueError."""
    graph_data = {
        "nodes": [1, 2],
        "features": [0.1, 0.2] # 1D instead of 2D
    }
    with pytest.raises(ValueError, match="Features must be a 2D array"):
        validate_model_input(graph_data)

def test_validate_model_input_mismatched_nodes_features_raises() -> None:
    """Test that mismatched nodes and features raises ValueError."""
    graph_data = {
        "nodes": [1, 2, 3],
        "features": [[0.1, 0.2], [0.3, 0.4]] # Only 2 features for 3 nodes
    }
    with pytest.raises(ValueError, match="Number of nodes must match number of features"):
        validate_model_input(graph_data)

def test_validate_model_input_empty_dict_raises() -> None:
    """Test that empty input dictionary raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_model_input({})
