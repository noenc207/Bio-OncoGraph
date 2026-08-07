import time
import json
import os
import psutil
import gc
try:
    import torch
except ImportError:
    torch = None

class ResourceProfiler:
    def __init__(self, model_name="model", seed=42, output_dir="."):
        self.model_name = model_name
        self.seed = seed
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        
        self.metrics = {
            "model": model_name,
            "seed": seed,
            "inference_time_s": 0.0,
            "peak_cpu_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "flops": 0,
            "params": 0
        }
        self.process = psutil.Process(os.getpid())
        self.start_cpu_mem = 0.0

    def start(self):
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        self.start_cpu_mem = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.time()

    def stop(self, flops=0, params=0):
        self.end_time = time.time()
        
        self.metrics["inference_time_s"] = self.end_time - self.start_time
        
        current_cpu_mem = self.process.memory_info().rss / (1024 * 1024)
        # Approximate peak CPU memory by max of start/end, in a real scenario threading might be used
        self.metrics["peak_cpu_memory_mb"] = max(current_cpu_mem - self.start_cpu_mem, 0)
        
        if torch is not None and torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.metrics["peak_gpu_memory_mb"] = peak_gpu
            
        self.metrics["flops"] = flops
        self.metrics["params"] = params
        
    def save(self, filename="compute_metrics.json"):
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {filepath}")

def track_resources(model_name="model", seed=42, output_dir="."):
    """Context manager or decorator alternative"""
    return ResourceProfiler(model_name=model_name, seed=seed, output_dir=output_dir)

if __name__ == "__main__":
    # Example usage
    profiler = ResourceProfiler("example_model", 42, "outputs")
    profiler.start()
    time.sleep(1) # mock inference
    profiler.stop(flops=1000000, params=500000)
    profiler.save()
