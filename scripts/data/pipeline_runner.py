import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from bio_oncograph.research.histograph.config import HistographConfig
from bio_oncograph.research.histograph.cell_graph import CellGraphBuilder, CellGraph
from bio_oncograph.research.histograph.tissue_graph import TissueGraphBuilder, TissueGraph
from bio_oncograph.research.histograph.region_graph import RegionGraphBuilder, RegionGraph

logger = logging.getLogger(__name__)

class PinnedMemoryWrapper:
    """Wrapper to pin memory for faster host-to-device transfer."""
    def __init__(self, data: np.ndarray):
        self.data = data
        self._pinned = False
    
    def pin(self) -> np.ndarray:
        """Simulates memory pinning by ensuring contiguous memory alignment."""
        self.data = np.ascontiguousarray(self.data)
        self._pinned = True
        return self.data
        
    @property
    def is_pinned(self) -> bool:
        return self._pinned

class AsyncPrefetcher:
    """Asynchronous prefetcher for data loading."""
    def __init__(self, data_iterator: Iterator[Any], prefetch_size: int = 2):
        self.data_iterator = data_iterator
        self.prefetch_size = prefetch_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=prefetch_size)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = asyncio.Event()

    async def _prefetch_task(self):
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            try:
                # Run the blocking next() call in an executor
                item = await loop.run_in_executor(self._executor, next, self.data_iterator)
                await self.queue.put(item)
            except StopIteration:
                await self.queue.put(None) # Sentinel value for end of iterator
                break
            except Exception as e:
                logger.error(f"Error in prefetch task: {e}")
                await self.queue.put(None)
                break

    async def start(self):
        self._task = asyncio.create_task(self._prefetch_task())

    async def stop(self):
        self._stop_event.set()
        self._executor.shutdown(wait=False)
        if hasattr(self, '_task') and not self._task.done():
            self._task.cancel()

    async def next(self):
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
    
    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.next()

@dataclass
class GraphPipelineResult:
    graph_type: str
    nodes_count: int
    edges_count: int

class GraphPipelineRunner:
    def __init__(self, prefetch_size: int = 2):
        self.prefetch_size = prefetch_size
        self.cell_builder = CellGraphBuilder()
        self.tissue_builder = TissueGraphBuilder()
        self.region_builder = RegionGraphBuilder()

    async def process_batch(self, batch: Dict[str, Any]) -> GraphPipelineResult:
        """Process a single batch of data asynchronously."""
        graph_type = batch.get("graph_type", "cell")
        config = batch.get("config", HistographConfig(graph_type=graph_type))
        
        # Apply pinned memory wrappers
        features = PinnedMemoryWrapper(batch["features"]).pin()
        
        if graph_type == "cell":
            centroids = PinnedMemoryWrapper(batch["centroids"]).pin()
            cell_types = batch.get("cell_types")
            if cell_types is not None:
                cell_types = PinnedMemoryWrapper(cell_types).pin()
            
            graph = self.cell_builder.build(
                centroids=centroids,
                features=features,
                config=config,
                cell_types=cell_types
            )
            return GraphPipelineResult(graph_type, len(graph.nodes), graph.metadata.get("edge_count", 0))
            
        elif graph_type == "tissue":
            centroids = PinnedMemoryWrapper(batch["centroids"]).pin()
            region_labels = batch.get("region_labels")
            if region_labels is not None:
                region_labels = PinnedMemoryWrapper(region_labels).pin()
                
            graph = self.tissue_builder.build(
                centroids=centroids,
                features=features,
                config=config,
                region_labels=region_labels
            )
            return GraphPipelineResult(graph_type, len(graph.nodes), graph.metadata.get("edge_count", 0))
            
        elif graph_type == "region":
            bboxes = PinnedMemoryWrapper(batch["bboxes"]).pin()
            region_types = batch.get("region_types")
            if region_types is not None:
                region_types = PinnedMemoryWrapper(region_types).pin()
                
            graph = self.region_builder.build(
                bboxes=bboxes,
                features=features,
                config=config,
                region_types=region_types
            )
            return GraphPipelineResult(graph_type, len(graph.nodes), graph.metadata.get("edge_count", 0))
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    async def run(self, data_iterator: Iterator[Dict[str, Any]]) -> List[GraphPipelineResult]:
        """Run the pipeline using async prefetching."""
        prefetcher = AsyncPrefetcher(data_iterator, prefetch_size=self.prefetch_size)
        await prefetcher.start()
        
        results = []
        try:
            async for batch in prefetcher:
                result = await self.process_batch(batch)
                results.append(result)
        finally:
            await prefetcher.stop()
            
        return results

def run_pipeline_sync(data_iterator: Iterator[Dict[str, Any]], prefetch_size: int = 2) -> List[GraphPipelineResult]:
    """Helper to run the async pipeline synchronously."""
    runner = GraphPipelineRunner(prefetch_size=prefetch_size)
    return asyncio.run(runner.run(data_iterator))
