import os
import sys
import shutil

def check_gpu():
    """Verify GPU availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Fallback to checking for nvidia-smi if torch is not installed
        return shutil.which('nvidia-smi') is not None

def check_disk_space(min_gb=10):
    """Verify sufficient disk space is available."""
    total, used, free = shutil.disk_usage(os.path.abspath(os.sep))
    free_gb = free / (1024 ** 3)
    return free_gb >= min_gb, free_gb

def check_experiment(exp_dir):
    """Check that necessary data directories exist for an experiment."""
    issues = []
    
    # Check embeddings
    emb_dir = os.path.join(exp_dir, "embeddings")
    if not os.path.exists(emb_dir) or not os.listdir(emb_dir):
        issues.append(f"Embeddings missing or empty in {emb_dir}")
        
    # Check graphs
    graph_dir = os.path.join(exp_dir, "graphs")
    if not os.path.exists(graph_dir) or not os.listdir(graph_dir):
        issues.append(f"Graphs missing or empty in {graph_dir}")
        
    return issues

def main(experiments_dir):
    print("Starting Pre-benchmark Checks...")
    
    # 1. GPU Check
    has_gpu = check_gpu()
    print(f"GPU Available: {has_gpu}")
    if not has_gpu:
        print("WARNING: No GPU detected. Benchmarking will be significantly slower.")
        
    # 2. Disk Check
    has_space, free_gb = check_disk_space()
    print(f"Disk Space: {free_gb:.2f} GB free")
    if not has_space:
        print(f"ERROR: Insufficient disk space ({free_gb:.2f} GB). Minimum required is 10 GB.")
        sys.exit(1)
        
    # 3. Experiment Data Check
    if not os.path.exists(experiments_dir):
        print(f"ERROR: Experiments directory '{experiments_dir}' does not exist.")
        sys.exit(1)
        
    all_issues = {}
    for exp in os.listdir(experiments_dir):
        exp_dir = os.path.join(experiments_dir, exp)
        if os.path.isdir(exp_dir):
            issues = check_experiment(exp_dir)
            if issues:
                all_issues[exp] = issues
                
    if all_issues:
        print("\nValidation Failed for some experiments:")
        for exp, issues in all_issues.items():
            print(f"  {exp}:")
            for issue in issues:
                print(f"    - {issue}")
        sys.exit(1)
        
    print("\nAll pre-benchmark checks passed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pre-benchmark checks.")
    parser.add_argument("--experiments_dir", default="experiments", help="Directory containing experiment data")
    args = parser.parse_args()
    main(args.experiments_dir)
