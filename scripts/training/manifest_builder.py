import os
import json
import hashlib
import argparse
from pathlib import Path

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def build_manifest(run_dir, output_file):
    run_path = Path(run_dir)
    artifacts = {}
    
    for root, _, files in os.walk(run_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.absolute() == Path(output_file).absolute():
                continue
                
            rel_path = file_path.relative_to(run_path)
            artifacts[str(rel_path)] = {
                "size_bytes": file_path.stat().st_size,
                "sha256": get_sha256(file_path)
            }
            
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifacts, f, indent=4)
        
    print(f"Manifest written to {output_file} with {len(artifacts)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=".", help="Directory containing run files")
    parser.add_argument("--output", type=str, default="artifacts.json", help="Output JSON file path")
    args = parser.parse_args()
    
    build_manifest(args.run_dir, args.output)
