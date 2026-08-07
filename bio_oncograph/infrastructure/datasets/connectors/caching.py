"""Local cache management for dataset connectors."""

import json
import os
import shutil
from typing import Any, Dict, Optional


class CacheManager:
    """Local cache manager supporting cache invalidation and versioning."""

    def __init__(self, cache_dir: str) -> None:
        """Initializes the CacheManager.

        Args:
            cache_dir: The directory to store cached data.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Loads cache metadata from disk."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Saves cache metadata to disk."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=4)

    def is_cached(self, dataset_name: str, version: str) -> bool:
        """Checks if a specific dataset version is cached.

        Args:
            dataset_name: The name of the dataset.
            version: The version of the dataset.

        Returns:
            True if cached, False otherwise.
        """
        key = f"{dataset_name}_{version}"
        return key in self._metadata and os.path.exists(self._metadata[key]['path'])

    def get_cache_path(self, dataset_name: str, version: str) -> Optional[str]:
        """Gets the path to the cached dataset.

        Args:
            dataset_name: The name of the dataset.
            version: The version of the dataset.

        Returns:
            The path to the cached dataset, or None if not cached.
        """
        if self.is_cached(dataset_name, version):
            return self._metadata[f"{dataset_name}_{version}"]['path']
        return None

    def register_cache(self, dataset_name: str, version: str, path: str) -> None:
        """Registers a new dataset cache.

        Args:
            dataset_name: The name of the dataset.
            version: The version of the dataset.
            path: The path where the dataset is cached.
        """
        key = f"{dataset_name}_{version}"
        self._metadata[key] = {'path': path, 'version': version}
        self._save_metadata()

    def invalidate_cache(self, dataset_name: str, version: Optional[str] = None) -> None:
        """Invalidates and removes cached data.

        Args:
            dataset_name: The name of the dataset.
            version: Optional version to invalidate. If None, all versions are invalidated.
        """
        keys_to_remove = []
        for key in self._metadata:
            if key.startswith(f"{dataset_name}_"):
                if version is None or key == f"{dataset_name}_{version}":
                    path = self._metadata[key]['path']
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._metadata[key]
        self._save_metadata()
