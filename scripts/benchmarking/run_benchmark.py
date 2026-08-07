import json
import logging
import os
import time
import argparse
import traceback
from typing import Any
from pathlib import Path

from bio_oncograph.benchmark.benchmark_config import BenchmarkConfig
from bio_oncograph.benchmark.benchmark_runner import BenchmarkRunner

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_pipeline(model_name: str, seed: int):
    """
    Mock pipeline getter.
    In a real scenario, this would initialize the model and set the random seed.
    """
    def dummy_pipeline(inputs: Any) -> Any:
        # Simulate some work
        time.sleep(0.01)
        if model_name == "error_model":
            raise ValueError(f"Simulated failure for {model_name}")
        return {"output": f"processed {inputs} with {model_name} at seed {seed}"}
    return dummy_pipeline

def get_dataset(dataset_name: str):
    """
    Mock dataset getter.
    Returns a dummy batch of inputs.
    """
    return [f"input_data_for_{dataset_name}"]

def append_to_failed_runs(failed_runs_file: str, failure_record: dict):
    """Appends a failure record to a JSON list in a file."""
    failed_runs = []
    if os.path.exists(failed_runs_file):
        try:
            with open(failed_runs_file, 'r', encoding='utf-8') as f:
                failed_runs = json.load(f)
        except json.JSONDecodeError:
            pass
            
    failed_runs.append(failure_record)
    
    with open(failed_runs_file, 'w', encoding='utf-8') as f:
        json.dump(failed_runs, f, indent=4)

def run_benchmarks(datasets, models, seeds, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    failed_runs_file = os.path.join(output_dir, "failed_runs.json")
    results_file = os.path.join(output_dir, "results.json")
    
    config = BenchmarkConfig(num_runs=5, warmup_runs=1, track_memory=True, validate_reproducibility=False)
    runner = BenchmarkRunner(config=config)
    
    all_results = []
    
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                logging.info(f"Running benchmark for Dataset: {dataset}, Model: {model}, Seed: {seed}")
                try:
                    # 1. Setup pipeline and data
                    pipeline = get_pipeline(model, seed)
                    inputs = get_dataset(dataset)
                    
                    # 2. Run benchmark
                    report = runner.run_benchmark(pipeline_callable=pipeline, inputs=inputs)
                    
                    # 3. Store results
                    result = {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "report": {
                            "latency_ms": report.latency_ms,
                            "memory_mb": report.memory_mb,
                            "throughput": report.throughput,
                            "is_deterministic": report.is_deterministic
                        }
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    logging.error(f"Failed run: Dataset={dataset}, Model={model}, Seed={seed}. Error: {str(e)}")
                    failure_record = {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    append_to_failed_runs(failed_runs_file, failure_record)
                    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
        
    logging.info(f"Benchmarking complete. Results saved to {results_file}")
    if os.path.exists(failed_runs_file):
        logging.info(f"Some runs failed. Check {failed_runs_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Loop Benchmark Runner")
    parser.add_argument("--datasets", type=str, nargs='+', default=["dataset_A", "dataset_B"])
    parser.add_argument("--models", type=str, nargs='+', default=["model_1", "model_2"])
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    
    args = parser.parse_args()
    run_benchmarks(args.datasets, args.models, args.seeds, args.output_dir)
