import json
import logging
import os
import subprocess
import sys
import platform
import pkg_resources
from datetime import datetime
from pathlib import Path

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit
    except Exception:
        return "unknown"

def get_package_versions():
    packages = {}
    for dist in pkg_resources.working_set:
        packages[dist.project_name] = dist.version
    return packages

def build_manifest(output_path, seeds, configurations):
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "os": os.name,
            "platform": platform.platform(),
            "python_version": sys.version
        },
        "git_commit": get_git_commit(),
        "versions": get_package_versions(),
        "seeds": seeds,
        "configurations": configurations
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Benchmark Manifest")
    parser.add_argument("--output", type=str, default="benchmark_manifest.json")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    
    # Dummy configurations for standalone execution
    configs = {
        "datasets": ["dataset_A", "dataset_B"],
        "models": ["model_1", "model_2"]
    }
    
    build_manifest(args.output, args.seeds, configs)
    print(f"Manifest written to {args.output}")
