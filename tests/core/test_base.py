"""
Unit tests for abstract base classes in bio_oncograph.core.base.

These tests verify that:
  - ABCs cannot be instantiated directly.
  - Concrete subclasses that implement the interface work correctly.
  - The BaseLoss.forward delegates to compute.
  - Type contracts hold across the inheritance hierarchy.
"""

from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor

from bio_oncograph.core.base import (
    BaseGraphPooling,
    BaseLoss,
    BaseMessagePassing,
    BaseModel,
    IGraphBuilder,
)
from bio_oncograph.core.data_models import (
    BioGraphData,
    BioGraphOutput,
    SurvivalTarget,
)
from bio_oncograph.core.typing import LossOutput


# ── Concrete stubs for testing ──────────────────────────────────────


class StubModel(BaseModel):
    """Minimal concrete implementation of BaseModel for testing."""

    def forward(self, data: BioGraphData) -> BioGraphOutput:
        batch_size = 1
        time_bins = 4
        logits = torch.zeros(batch_size, time_bins)
        curves = torch.ones(batch_size, time_bins)
        return BioGraphOutput(hazard_logits=logits, survival_curves=curves)


class StubPooling(BaseGraphPooling):
    """Minimal concrete implementation of BaseGraphPooling for testing."""

    def pool(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return x, edge_index, torch.ones(x.size(0), dtype=torch.bool), batch


class StubMessagePassing(BaseMessagePassing):
    """Minimal concrete implementation of BaseMessagePassing for testing."""

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        x_parent: Optional[Tensor] = None,
        cluster_map: Optional[Tensor] = None,
    ) -> Tensor:
        return x


class StubLoss(BaseLoss):
    """Minimal concrete implementation of BaseLoss for testing."""

    def compute(
        self,
        pred: BioGraphOutput,
        target: SurvivalTarget,
    ) -> LossOutput:
        zero = torch.tensor(0.0)
        return LossOutput(total_loss=zero, nll=zero, ranking=zero, topo=zero)


class StubGraphBuilder(IGraphBuilder):
    """Minimal concrete implementation of IGraphBuilder for testing."""

    def build(self, features: Tensor, coords: Tensor, k: int) -> BioGraphData:
        num_nodes = features.size(0)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        return BioGraphData(x=features, pos=coords, edge_index=edge_index)


# ── Tests ───────────────────────────────────────────────────────────


def test_base_model_is_abstract() -> None:
    """BaseModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_base_graph_pooling_is_abstract() -> None:
    """BaseGraphPooling cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseGraphPooling()  # type: ignore[abstract]


def test_base_message_passing_is_abstract() -> None:
    """BaseMessagePassing cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseMessagePassing()  # type: ignore[abstract]


def test_base_loss_is_abstract() -> None:
    """BaseLoss cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseLoss()  # type: ignore[abstract]


def test_igraph_builder_is_abstract() -> None:
    """IGraphBuilder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IGraphBuilder()  # type: ignore[abstract]


def test_stub_model_forward() -> None:
    """Concrete model produces correctly shaped BioGraphOutput."""
    model = StubModel()
    data = BioGraphData(
        x=torch.randn(5, 16),
        pos=torch.randn(5, 2),
        edge_index=torch.randint(0, 5, (2, 8)),
    )
    output = model(data)
    assert isinstance(output, BioGraphOutput)
    assert output.hazard_logits.shape == (1, 4)
    assert output.survival_curves.shape == (1, 4)


def test_stub_pooling_returns_correct_tuple() -> None:
    """Concrete pooling returns the 4-tuple contract."""
    pool = StubPooling()
    x = torch.randn(10, 32)
    edge_index = torch.randint(0, 10, (2, 15))
    batch = torch.zeros(10, dtype=torch.long)

    x_out, ei_out, mask, batch_out = pool.pool(x, edge_index, batch)
    assert x_out.shape == x.shape
    assert mask.shape == (10,)


def test_stub_message_passing_identity() -> None:
    """Concrete MP returns features unchanged (identity stub)."""
    mp = StubMessagePassing()
    x = torch.randn(6, 16)
    edge_index = torch.randint(0, 6, (2, 10))
    out = mp(x, edge_index)
    assert torch.equal(out, x)


def test_base_loss_forward_delegates_to_compute() -> None:
    """BaseLoss.forward must call compute and return the same result."""
    loss_fn = StubLoss()
    pred = BioGraphOutput(
        hazard_logits=torch.zeros(1, 4),
        survival_curves=torch.ones(1, 4),
    )
    target = SurvivalTarget(
        time_bins=torch.tensor([2]),
        event_indicators=torch.tensor([True]),
    )
    result = loss_fn(pred, target)
    assert isinstance(result, LossOutput)
    assert result.total_loss.item() == 0.0


def test_stub_graph_builder_returns_biograph_data() -> None:
    """Concrete graph builder produces a BioGraphData instance."""
    builder = StubGraphBuilder()
    features = torch.randn(20, 64)
    coords = torch.randn(20, 2)
    data = builder.build(features, coords, k=5)
    assert isinstance(data, BioGraphData)
    assert data.x is not None
    assert data.x.shape == (20, 64)
