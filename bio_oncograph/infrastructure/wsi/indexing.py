"""Indexing module for Whole Slide Image patches."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PatchIndex:
    """Maintains a mapping of patch coordinates to index IDs."""
    
    def __init__(self) -> None:
        """Initialize the patch index."""
        self._index: Dict[Tuple[str, int, int, int], str] = {}
        self._slide_map: Dict[str, List[Tuple[int, int, int]]] = {}
        
    def add(self, patch_metadata: Any) -> str:
        """Add a patch to the index.
        
        Args:
            patch_metadata: Metadata object containing slide_id, x, y, level.
            
        Returns:
            The generated index ID.
        """
        key = (
            patch_metadata.slide_id, 
            patch_metadata.x, 
            patch_metadata.y, 
            patch_metadata.level
        )
        if key in self._index:
            logger.warning("Patch %s already indexed.", key)
            return self._index[key]
            
        index_id = str(uuid.uuid4())
        self._index[key] = index_id
        
        if patch_metadata.slide_id not in self._slide_map:
            self._slide_map[patch_metadata.slide_id] = []
        self._slide_map[patch_metadata.slide_id].append(
            (patch_metadata.x, patch_metadata.y, patch_metadata.level)
        )
        
        return index_id
        
    def lookup(self, slide_id: str, x: int, y: int, level: int) -> Optional[str]:
        """Look up a patch index ID by its coordinates.
        
        Args:
            slide_id: Slide identifier.
            x: X coordinate.
            y: Y coordinate.
            level: Magnification level.
            
        Returns:
            The index ID if found, else None.
        """
        return self._index.get((slide_id, x, y, level))
        
    def list_patches(self, slide_id: str) -> List[Tuple[int, int, int]]:
        """List all patches for a given slide.
        
        Args:
            slide_id: Slide identifier.
            
        Returns:
            List of (x, y, level) tuples.
        """
        return self._slide_map.get(slide_id, []).copy()
        
    def size(self) -> int:
        """Get the total number of indexed patches.
        
        Returns:
            Number of indexed patches.
        """
        return len(self._index)
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the index to a dictionary.
        
        Returns:
            Dictionary representation of the index.
        """
        str_index = {
            f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v 
            for k, v in self._index.items()
        }
        return {
            "index": str_index,
            "slide_map": self._slide_map
        }
        
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize the index from a dictionary.
        
        Args:
            data: Dictionary representation of the index.
        """
        self._index.clear()
        self._slide_map.clear()
        
        str_index = data.get("index", {})
        for k_str, v in str_index.items():
            parts = k_str.split("_")
            if len(parts) >= 4:
                slide_id = "_".join(parts[:-3])
                x = int(parts[-3])
                y = int(parts[-2])
                level = int(parts[-1])
                self._index[(slide_id, x, y, level)] = v
                
        self._slide_map = data.get("slide_map", {})
