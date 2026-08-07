"""Tools for creating PyG Batches from graph data."""

from typing import Any, Dict, List
import torch_geometric
from torch_geometric.data import Data, Batch

from bio_oncograph.infrastructure.pyg.adapters import GraphAdapter

class BatchBuilder:
    """Builder for PyG Batch objects."""

    def __init__(self) -> None:
        """Initializes the BatchBuilder."""
        self.adapter = GraphAdapter()

    def build_from_dicts(self, graph_dicts: List[Dict[str, Any]]) -> Batch:
        """Builds a Batch from a list of standard graph dictionaries.
        
        Args:
            graph_dicts: List of graph dictionaries.
            
        Returns:
            A PyG Batch object.
        """
        return self.adapter.to_pyg_batch(graph_dicts)

    def build_from_data_list(self, data_list: List[Data]) -> Batch:
        """Builds a Batch from a list of PyG Data objects.
        
        Args:
            data_list: List of PyG Data objects.
            
        Returns:
            A PyG Batch object.
        """
        return Batch.from_data_list(data_list)
