"""Adapters between framework internal graph representations and PyTorch Geometric Data."""

from typing import Any, Dict, List
import torch
import torch_geometric
from torch_geometric.data import Data, Batch

class GraphAdapter:
    """Adapter for converting to and from PyG Data objects."""

    def __init__(self) -> None:
        """Initializes the GraphAdapter."""
        pass

    def to_pyg(self, graph_dict: Dict[str, Any]) -> Data:
        """Converts a standard graph dictionary to a PyG Data object.
        
        Args:
            graph_dict: Dictionary containing 'x', 'edge_index', 'edge_attr', 'y'.
            
        Returns:
            A torch_geometric.data.Data object.
        """
        x = torch.tensor(graph_dict.get('x', []), dtype=torch.float)
        edge_index = torch.tensor(graph_dict.get('edge_index', [[], []]), dtype=torch.long)
        edge_attr = torch.tensor(graph_dict.get('edge_attr', []), dtype=torch.float) if 'edge_attr' in graph_dict else None
        y = torch.tensor(graph_dict.get('y', []), dtype=torch.float) if 'y' in graph_dict else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def from_pyg(self, pyg_data: Data) -> Dict[str, Any]:
        """Converts a PyG Data object back to a standard dictionary.
        
        Args:
            pyg_data: A torch_geometric.data.Data object.
            
        Returns:
            Dictionary representation of the graph.
        """
        result = {
            'x': pyg_data.x.tolist() if pyg_data.x is not None else [],
            'edge_index': pyg_data.edge_index.tolist() if pyg_data.edge_index is not None else [[], []]
        }
        if pyg_data.edge_attr is not None:
            result['edge_attr'] = pyg_data.edge_attr.tolist()
        if pyg_data.y is not None:
            result['y'] = pyg_data.y.tolist()
        return result
        
    def to_pyg_batch(self, graph_dicts: List[Dict[str, Any]]) -> Batch:
        """Converts a list of graph dictionaries to a PyG Batch object.
        
        Args:
            graph_dicts: List of graph dictionaries.
            
        Returns:
            A torch_geometric.data.Batch object.
        """
        data_list = [self.to_pyg(g) for g in graph_dicts]
        return Batch.from_data_list(data_list)
