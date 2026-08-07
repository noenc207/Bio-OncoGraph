import os
import json
import pickle
from typing import Any, Dict

class DatasetCacheManager:
    """Manages caching of metadata, WSI, features, and graphs."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.metadata_dir = os.path.join(self.cache_dir, "metadata")
        self.wsi_dir = os.path.join(self.cache_dir, "wsi")
        self.feature_dir = os.path.join(self.cache_dir, "features")
        self.graph_dir = os.path.join(self.cache_dir, "graphs")
        
        for d in [self.metadata_dir, self.wsi_dir, self.feature_dir, self.graph_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.memory_cache: Dict[str, Any] = {}

    def _get_path(self, cache_type: str, key: str, ext: str = ".pkl") -> str:
        type_to_dir = {
            "metadata": self.metadata_dir,
            "wsi": self.wsi_dir,
            "feature": self.feature_dir,
            "graph": self.graph_dir
        }
        return os.path.join(type_to_dir[cache_type], f"{key}{ext}")

    def load_or_create(self, cache_type: str, key: str, create_fn, lazy: bool = False) -> Any:
        """Loads data from cache or creates it using the provided function."""
        mem_key = f"{cache_type}_{key}"
        if mem_key in self.memory_cache:
            return self.memory_cache[mem_key]
            
        ext = ".json" if cache_type == "metadata" else ".pkl"
        path = self._get_path(cache_type, key, ext)
        
        if os.path.exists(path):
            if lazy:
                return path
            if ext == ".json":
                with open(path, "r") as f:
                    data = json.load(f)
            else:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            self.memory_cache[mem_key] = data
            return data
            
        data = create_fn()
        self.save(cache_type, key, data)
        self.memory_cache[mem_key] = data
        return data

    def save(self, cache_type: str, key: str, data: Any) -> None:
        """Saves data to cache."""
        ext = ".json" if cache_type == "metadata" else ".pkl"
        path = self._get_path(cache_type, key, ext)
        
        if ext == ".json":
            with open(path, "w") as f:
                json.dump(data, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)
                
    def clear_cache(self) -> None:
        """Clears the in-memory cache."""
        self.memory_cache.clear()
