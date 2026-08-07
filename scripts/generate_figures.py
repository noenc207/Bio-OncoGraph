import os
import json
import hashlib
from typing import Dict, Any, Callable

class FigureCache:
    def __init__(self, cache_file: str = 'figure_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def get(self, key: str) -> str:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value
        self._save_cache()

    def generate(self, figure_id: str, manifest: Dict[str, Any], generator_func: Callable[..., str]) -> str:
        manifest_str = json.dumps(manifest, sort_keys=True)
        key = hashlib.md5(f"{figure_id}_{manifest_str}".encode('utf-8')).hexdigest()
        cached = self.get(key)
        if cached and os.path.exists(cached):
            return cached
        
        output_path = f"outputs/figures/{figure_id}_{key}.png"
        generator_func(output_path, manifest)
        
        trace_path = f"outputs/figures/{figure_id}_{key}_trace.json"
        with open(trace_path, 'w') as f:
            json.dump({
                'figure_id': figure_id,
                'output_path': output_path,
                'manifest': manifest
            }, f, indent=4)
            
        self.set(key, output_path)
        return output_path

def generate_umap(output_path: str, manifest: Dict[str, Any]):
    with open(output_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

def generate_gradcam(output_path: str, manifest: Dict[str, Any]):
    with open(output_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

def generate_km(output_path: str, manifest: Dict[str, Any]):
    with open(output_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

def main():
    cache = FigureCache()
    generators = {
        'UMAP': generate_umap,
        'GradCAM': generate_gradcam,
        'KM': generate_km,
    }
    
    os.makedirs('outputs/figures', exist_ok=True)
    
    manifest = {
        'experiment_id': 'exp_001',
        'model_version': 'v1.0.0',
        'hyperparameters': {'lr': 0.001, 'batch_size': 32},
        'dataset': 'dataset_a'
    }
    
    for f_id, func in generators.items():
        path = cache.generate(f_id, manifest, func)
        print(f"Generated {f_id} at {path} with traceability")

if __name__ == '__main__':
    main()
