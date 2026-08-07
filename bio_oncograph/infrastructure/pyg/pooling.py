"""Wrappers for PyTorch Geometric pooling operations."""

import torch
import torch_geometric
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    TopKPooling,
    SAGPooling
)
from torch import Tensor
from typing import Optional, Tuple, Union


def apply_global_mean_pool(x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None) -> Tensor:
    """Apply global mean pooling.

    Args:
        x (Tensor): Node feature matrix.
        batch (Optional[Tensor], optional): Batch vector, which assigns each node to a specific graph. Defaults to None.
        size (Optional[int], optional): The number of graphs. Defaults to None.

    Returns:
        Tensor: Graph-level feature matrix.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return global_mean_pool(x, batch, size)


def apply_global_max_pool(x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None) -> Tensor:
    """Apply global max pooling.

    Args:
        x (Tensor): Node feature matrix.
        batch (Optional[Tensor], optional): Batch vector, which assigns each node to a specific graph. Defaults to None.
        size (Optional[int], optional): The number of graphs. Defaults to None.

    Returns:
        Tensor: Graph-level feature matrix.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return global_max_pool(x, batch, size)


def apply_global_add_pool(x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None) -> Tensor:
    """Apply global add pooling.

    Args:
        x (Tensor): Node feature matrix.
        batch (Optional[Tensor], optional): Batch vector, which assigns each node to a specific graph. Defaults to None.
        size (Optional[int], optional): The number of graphs. Defaults to None.

    Returns:
        Tensor: Graph-level feature matrix.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return global_add_pool(x, batch, size)


class TopKPoolingExt(torch.nn.Module):
    """Wrapper for TopKPooling."""

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5) -> None:
        """Initialize TopKPoolingExt.

        Args:
            in_channels (int): Size of each input sample.
            ratio (Union[float, int], optional): The ratio of nodes to keep. Defaults to 0.5.
        """
        super().__init__()
        self.pool = TopKPooling(in_channels=in_channels, ratio=ratio)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        """Forward pass for TopKPooling.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity matrix.
            edge_attr (Optional[Tensor], optional): Edge feature matrix. Defaults to None.
            batch (Optional[Tensor], optional): Batch vector. Defaults to None.

        Returns:
            Tuple: Output features, edge indices, edge attributes, batch, permutation, and score.
        """
        return self.pool(x, edge_index, edge_attr, batch)


class SAGPoolingExt(torch.nn.Module):
    """Wrapper for SAGPooling."""

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5) -> None:
        """Initialize SAGPoolingExt.

        Args:
            in_channels (int): Size of each input sample.
            ratio (Union[float, int], optional): The ratio of nodes to keep. Defaults to 0.5.
        """
        super().__init__()
        self.pool = SAGPooling(in_channels=in_channels, ratio=ratio)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        """Forward pass for SAGPooling.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity matrix.
            edge_attr (Optional[Tensor], optional): Edge feature matrix. Defaults to None.
            batch (Optional[Tensor], optional): Batch vector. Defaults to None.

        Returns:
            Tuple: Output features, edge indices, edge attributes, batch, permutation, and score.
        """
        return self.pool(x, edge_index, edge_attr, batch)
