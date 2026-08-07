"""
Abstract base classes defining the interface contracts for Bio-OncoGraph modules.

Every concrete module (PhenoGraph, TopoPool, BioMP, CalTopoSurv) MUST inherit
from one of these ABCs. This guarantees Liskov Substitution and enables the
Registry/Factory pattern to instantiate modules interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

from bio_oncograph.core.data_models import BioGraphData, BioGraphOutput, SurvivalTarget
from bio_oncograph.core.typing import LossOutput

class BaseModel(ABC):
    """
    Abstract base class for the top-level Bio-OncoGraph model.

    Subclasses must implement ``forward`` to accept a ``BioGraphData``
    object and return a ``BioGraphOutput``.  The forward method MUST be
    stateless: it must NOT store intermediate tensors on ``self``.
    """

    @abstractmethod
    def forward(self, data: BioGraphData) -> BioGraphOutput:
        """
        Run the full forward pass of the model.

        Args:
            data: A ``BioGraphData`` instance containing node features,
                  edge connectivity, spatial coordinates, and optional
                  survival targets.

        Returns:
            A ``BioGraphOutput`` containing hazard logits, survival curves,
            and optional explainability tensors.
        """
        ...


class BaseGraphPooling(ABC):
    """
    Abstract base class for graph pooling operators.

    Subclasses (e.g. TopoPool, DiffPool) must implement ``pool`` to
    reduce the number of nodes while preserving topological structure.
    """

    @abstractmethod
    def pool(
        self,
        x: Any,
        edge_index: Any,
        batch: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Pool (coarsen) the graph.

        Args:
            x: Node feature matrix of shape ``(N, D)``.
            edge_index: Edge connectivity of shape ``(2, E)``.
            batch: Batch assignment vector of shape ``(N,)``.

        Returns:
            A 4-tuple of:
                - ``x_pool``: Pooled node features ``(N', D)``.
                - ``edge_index_pool``: Pooled edge connectivity ``(2, E')``.
                - ``pool_mask``: Boolean or score mask indicating retained
                  nodes ``(N,)``.
                - ``batch_pool``: Updated batch vector ``(N',)``.

        Raises:
            TopologyCollapseException: If all nodes are removed.
        """
        ...


class BaseMessagePassing(ABC):
    """
    Abstract base class for message-passing layers.

    Subclasses (e.g. BioMP) implement a single round of neighbourhood
    aggregation, optionally incorporating cross-scale information from
    a parent graph.
    """

    @abstractmethod
    def forward(
        self,
        x: Any,
        edge_index: Any,
        x_parent: Optional[Any] = None,
        cluster_map: Optional[Any] = None,
    ) -> Any:
        """
        Perform one round of message passing.

        Args:
            x: Node feature matrix ``(N, D_in)``.
            edge_index: Edge connectivity ``(2, E)``.
            x_parent: Optional parent-level node features for cross-scale
                      messaging ``(N_parent, D_parent)``.
            cluster_map: Optional mapping from child to parent nodes
                         ``(N,)``.

        Returns:
            Updated node feature matrix ``(N, D_out)``.
        """
        ...


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    Subclasses (e.g. CalTopoSurv) must implement ``compute`` to return
    a structured ``LossOutput`` containing the total loss and its
    individual components.

    .. warning::
        Loss computation MUST be performed in ``float32`` even when AMP
        is active.  Implementations should decorate ``compute`` with
        ``@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)`` or
        manually cast inputs.
    """

    @abstractmethod
    def compute(
        self,
        pred: BioGraphOutput,
        target: SurvivalTarget,
    ) -> LossOutput:
        """
        Compute the composite loss.

        Args:
            pred: Model predictions containing hazard logits and
                  survival curves.
            target: Ground-truth survival targets.

        Returns:
            A ``LossOutput`` named-tuple with ``total_loss``, ``nll``,
            ``ranking``, and ``topo`` components.

        Raises:
            NaNGradientException: If any component evaluates to NaN.
        """
        ...

    def forward(
        self,
        pred: BioGraphOutput,
        target: SurvivalTarget,
    ) -> LossOutput:
        """
        Delegate to ``compute`` so the loss can be used as a standard
        ``nn.Module`` in the training loop.

        Args:
            pred: Model predictions.
            target: Ground-truth survival targets.

        Returns:
            A ``LossOutput`` named-tuple.
        """
        return self.compute(pred, target)


class IGraphBuilder(ABC):
    """
    Abstract interface for graph construction strategies.

    Subclasses implement different algorithms to convert raw patch
    features and spatial coordinates into a ``BioGraphData`` instance.
    This interface was introduced following the Final Architecture Audit
    recommendation to decouple graph construction from the model layer
    (SOLID — Dependency Inversion Principle).
    """

    @abstractmethod
    def build(
        self,
        features: Any,
        coords: Any,
        k: int,
    ) -> BioGraphData:
        """
        Construct a graph from patch-level features and coordinates.

        Args:
            features: Patch feature matrix ``(N, D)``.
            coords: Spatial coordinates ``(N, 2)``.
            k: Number of nearest neighbours for edge construction.

        Returns:
            A ``BioGraphData`` instance with ``x``, ``pos``, and
            ``edge_index`` populated.

        Raises:
            InvalidGraphTopologyException: If the resulting graph is
                degenerate (e.g. zero edges).
        """
        ...
