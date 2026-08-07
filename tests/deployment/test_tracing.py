import pytest
from typing import List, Dict, Any

class DistributedTracer:
    def __init__(self) -> None:
        self.active = False
        self.graph: List[str] = []

    def start_trace(self) -> None:
        self.active = True

    def add_node(self, node: str) -> None:
        if self.active:
            self.graph.append(node)

    def extract_lineage(self) -> List[str]:
        return self.graph.copy()

@pytest.fixture
def tracer() -> DistributedTracer:
    return DistributedTracer()

@pytest.mark.parametrize("nodes_count", range(30))
def test_tracer_active_lineage_extraction(tracer: DistributedTracer, nodes_count: int) -> None:
    """Test distributed tracer activity and extract lineage graph."""
    tracer.start_trace()
    for i in range(nodes_count):
        tracer.add_node(f"node_{i}")
    lineage = tracer.extract_lineage()
    assert len(lineage) == nodes_count
    if nodes_count > 0:
        assert lineage[-1] == f"node_{nodes_count - 1}"

@pytest.mark.parametrize("idx", range(10))
def test_tracer_inactive(tracer: DistributedTracer, idx: int) -> None:
    """Test tracer when inactive."""
    tracer.add_node(f"node_{idx}")
    assert len(tracer.extract_lineage()) == 0
