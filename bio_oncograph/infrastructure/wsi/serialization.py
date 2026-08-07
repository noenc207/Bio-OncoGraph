"""Serialization module for patches and indices."""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PatchSerializer:
    """Save and load patch metadata and coordinate mappings."""
    
    VERSION = "1.0.0"
    
    @classmethod
    def save(cls, metadata_list: List[Any], path: str) -> None:
        """Save a list of patch metadata to JSON.
        
        Args:
            metadata_list: List of metadata objects/dicts.
            path: Path to save the JSON file.
        """
        try:
            data_to_serialize = [
                m if isinstance(m, dict) else m.__dict__ 
                for m in metadata_list
            ]
            
            output = {
                "version": cls.VERSION,
                "metadata": data_to_serialize
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            logger.info("Successfully saved patch metadata to %s", path)
        except Exception as e:
            logger.error("Failed to save patch metadata: %s", e)
            raise

    @classmethod
    def load(cls, path: str) -> List[Dict[str, Any]]:
        """Load patch metadata from JSON.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            List of metadata dictionaries.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if data.get("version") != cls.VERSION:
                logger.warning(
                    "Version mismatch: file %s, serializer %s",
                    data.get("version"), cls.VERSION
                )
                
            return data.get("metadata", [])
        except Exception as e:
            logger.error("Failed to load patch metadata: %s", e)
            raise


class SlideIndexSerializer:
    """Save and load PatchIndex objects to/from JSON."""
    
    VERSION = "1.0.0"
    
    @classmethod
    def save(cls, index: Any, path: str) -> None:
        """Save a PatchIndex to JSON.
        
        Args:
            index: PatchIndex instance.
            path: Path to save the JSON file.
        """
        try:
            output = {
                "version": cls.VERSION,
                "index_data": index.to_dict()
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            logger.info("Successfully saved index to %s", path)
        except Exception as e:
            logger.error("Failed to save slide index: %s", e)
            raise

    @classmethod
    def load(cls, index: Any, path: str) -> None:
        """Load a PatchIndex from JSON into an existing object.
        
        Args:
            index: PatchIndex instance to populate.
            path: Path to the JSON file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if data.get("version") != cls.VERSION:
                logger.warning(
                    "Version mismatch: file %s, serializer %s",
                    data.get("version"), cls.VERSION
                )
                
            index.from_dict(data.get("index_data", {}))
            logger.info("Successfully loaded index from %s", path)
        except Exception as e:
            logger.error("Failed to load slide index: %s", e)
            raise
