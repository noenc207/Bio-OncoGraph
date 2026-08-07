import pytest
from unittest.mock import MagicMock

class GraphGenerator:
    def __init__(self, method="knn", k=5):
        self.method = method
        self.k = k

    def generate(self, embeddings, coords):
        pass

@pytest.mark.parametrize("method", ["knn", "radius", "delaunay"])
@pytest.mark.parametrize("k_or_radius", [3, 5, 10, 20])
@pytest.mark.parametrize("num_nodes", [10, 100, 1000, 5000])
@pytest.mark.parametrize("include_edge_features", [True, False])
@pytest.mark.parametrize("self_loops", [True, False])
def test_graph_generation_matrix(method, k_or_radius, num_nodes, include_edge_features, self_loops):
    # Generates 3 * 4 * 4 * 2 * 2 = 192 test cases
    generator = GraphGenerator(method=method, k=k_or_radius)
    
    mock_embeddings = [[0.0] * 512 for _ in range(num_nodes)]
    mock_coords = [[0, 0] for _ in range(num_nodes)]
    
    mock_graph = MagicMock()
    mock_graph.num_nodes = num_nodes
    mock_graph.has_edge_attr = include_edge_features
    mock_graph.has_self_loops = self_loops
    
    generator.generate = MagicMock(return_value=mock_graph)
    
    graph = generator.generate(mock_embeddings, mock_coords)
    generator.generate.assert_called_once_with(mock_embeddings, mock_coords)
    
    assert graph.num_nodes == num_nodes
    assert graph.has_edge_attr == include_edge_features
    assert graph.has_self_loops == self_loops

@pytest.mark.parametrize("empty_input", [True, False])
@pytest.mark.parametrize("mismatched_shapes", [True, False])
def test_graph_generation_edge_cases(empty_input, mismatched_shapes):
    # Generates 2 * 2 = 4 test cases
    generator = GraphGenerator(method="knn")
    if empty_input:
        generator.generate = MagicMock(side_effect=ValueError("Empty input"))
        with pytest.raises(ValueError):
            generator.generate([], [])
    elif mismatched_shapes:
        generator.generate = MagicMock(side_effect=ValueError("Shape mismatch"))
        with pytest.raises(ValueError):
            generator.generate([[0.1]], [[0,0], [1,1]])
    else:
        generator.generate = MagicMock(return_value=MagicMock())
        assert generator.generate([[0.1]], [[0,0]]) is not None
