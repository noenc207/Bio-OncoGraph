"""Message passing abstraction wrapping PyG's MessagePassing."""

import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from typing import Any, Optional, Tuple, Union
from torch import Tensor


class BaseMessagePassingExt(MessagePassing):
    """Abstraction wrapping PyG's MessagePassing class for integration with framework Encoders.
    
    Args:
        aggr (str, optional): The aggregation scheme to use (e.g., "add", "mean", "max").
            Defaults to "add".
        flow (str, optional): The flow direction of message passing (e.g., "source_to_target").
            Defaults to "source_to_target".
        node_dim (int, optional): The axis along which to propagate. Defaults to -2.
    """

    def __init__(
        self,
        aggr: str = "add",
        flow: str = "source_to_target",
        node_dim: int = -2,
        **kwargs: Any
    ) -> None:
        """Initialize the BaseMessagePassingExt."""
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim, **kwargs)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Tensor:
        """Perform a single message passing step.

        Args:
            x (Union[Tensor, Tuple[Tensor, Tensor]]): Node feature matrix.
            edge_index (Tensor): Graph connectivity matrix.
            edge_attr (Optional[Tensor], optional): Edge feature matrix. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Updated node feature matrix.
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, **kwargs)

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        """Construct messages to node i.

        Args:
            x_j (Tensor): The features of the neighboring nodes.
            edge_attr (Optional[Tensor], optional): The edge features. Defaults to None.

        Returns:
            Tensor: The constructed messages.
        """
        if edge_attr is not None:
            return x_j + edge_attr
        return x_j

    def update(self, inputs: Tensor, x: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Update node features.

        Args:
            inputs (Tensor): The aggregated messages.
            x (Union[Tensor, Tuple[Tensor, Tensor]]): The initial node features.

        Returns:
            Tensor: The updated node features.
        """
        return inputs
