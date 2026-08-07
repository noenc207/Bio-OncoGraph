import torch
import torch.nn as nn
import time
import json
from typing import Dict, Any, Tuple
import os
from torch.profiler import profile, ProfilerActivity

class ComputeAnalyzer:
    """
    Auto-calculates FLOPs, Parameter counts, GPU memory usage, and inference time.
    """
    def __init__(self, model: nn.Module, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def measure_inference_time(self, dummy_input: torch.Tensor, num_iterations: int = 100) -> float:
        """Measures average inference time over num_iterations."""
        dummy_input = dummy_input.to(self.device)
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)
                
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
            
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                self.model(dummy_input)
                
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
            
        end_time = time.time()
        return (end_time - start_time) / num_iterations

    def get_gpu_memory_usage(self) -> float:
        """Gets peak GPU memory usage in MB."""
        if not self.device.startswith('cuda'):
            return 0.0
        torch.cuda.reset_peak_memory_stats(self.device)
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def estimate_flops(self, dummy_input: torch.Tensor) -> int:
        """Estimates total FLOPs using PyTorch profiler."""
        dummy_input = dummy_input.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
                self.model(dummy_input)
                
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops > 0)
        return total_flops

    def generate_report(self, dummy_input: torch.Tensor, output_path: str = "compute_report.json") -> Dict[str, Any]:
        """Generates compute report and exports to JSON."""
        params = self.count_parameters()
        inf_time = self.measure_inference_time(dummy_input)
        mem_usage = self.get_gpu_memory_usage()
        
        try:
            flops = self.estimate_flops(dummy_input)
        except Exception:
            flops = -1
            
        report = {
            "parameters_count": params,
            "inference_time_seconds": inf_time,
            "gpu_memory_usage_mb": mem_usage,
            "flops": flops
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
            
        return report
