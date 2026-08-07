import os
import json
import torch
import argparse
import psutil

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
except OSError:
    HAS_OPENSLIDE = False

def get_gpu_memory():
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "device_id": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2)
            })
    return gpus

def check_file(path):
    if not path:
        return {"status": "not_provided"}
    if os.path.exists(path):
        return {"status": "found", "path": path}
    return {"status": "missing", "path": path}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_manifest', type=str, default='dataset_manifest.csv')
    parser.add_argument('--feature_manifest', type=str, default='feature_manifest.csv')
    parser.add_argument('--graph_manifest', type=str, default='graph_manifest.csv')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='preflight_report.json')
    args = parser.parse_args()

    report = {
        "manifests": {
            "dataset": check_file(args.dataset_manifest),
            "feature": check_file(args.feature_manifest),
            "graph": check_file(args.graph_manifest)
        },
        "config": check_file(args.config),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": get_gpu_memory()
        },
        "openslide": {
            "available": HAS_OPENSLIDE
        },
        "system_memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Preflight check completed. Report saved to {args.output}")

if __name__ == "__main__":
    main()
