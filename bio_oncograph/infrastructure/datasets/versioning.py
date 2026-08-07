import os
import hashlib
import json
import time
from typing import Dict, Any

class DatasetVersionManager:
    """Manages dataset versioning, including checksums and manifests."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.manifest_path = os.path.join(dataset_dir, "manifest.json")

    def calculate_sha256(self, filepath: str) -> str:
        """Calculates the SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_manifest(self, release_version: str) -> Dict[str, Any]:
        """Generates a manifest of all files in the dataset directory."""
        manifest = {
            "release_version": release_version,
            "timestamp": time.time(),
            "files": {}
        }
        
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file == "manifest.json":
                    continue
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.dataset_dir)
                manifest["files"][rel_path] = {
                    "sha256": self.calculate_sha256(filepath),
                    "size": os.path.getsize(filepath)
                }
                
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        return manifest

    def verify_integrity(self) -> bool:
        """Verifies dataset integrity using the manifest."""
        if not os.path.exists(self.manifest_path):
            return False
            
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
            
        for rel_path, info in manifest["files"].items():
            filepath = os.path.join(self.dataset_dir, rel_path)
            if not os.path.exists(filepath):
                return False
            if self.calculate_sha256(filepath) != info["sha256"]:
                return False
            if os.path.getsize(filepath) != info["size"]:
                return False
                
        return True
