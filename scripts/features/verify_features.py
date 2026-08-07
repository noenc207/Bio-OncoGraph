import os
import argparse
import logging
import h5py
import lmdb
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def verify_tensor(tensor, expected_dim=None):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    
    issues = []
    if torch.isnan(tensor).any():
        issues.append("Contains NaN")
    if torch.isinf(tensor).any():
        issues.append("Contains Inf")
    if expected_dim is not None and tensor.dim() != expected_dim:
        issues.append(f"Dimension mismatch: expected {expected_dim}, got {tensor.dim()}")
        
    return issues

def verify_pt(directory, expected_dim):
    paths = list(Path(directory).glob('*.pt'))
    logger.info(f"Found {len(paths)} .pt files.")
    
    corrupt = 0
    issues_found = 0
    for p in paths:
        try:
            tensor = torch.load(p, map_location='cpu')
            issues = verify_tensor(tensor, expected_dim)
            if issues:
                logger.warning(f"Issues in {p.name}: {', '.join(issues)}")
                issues_found += 1
        except Exception as e:
            logger.error(f"Corrupt tensor file {p.name}: {e}")
            corrupt += 1
            
    logger.info(f"PT Verification complete. Corrupt: {corrupt}, Content Issues: {issues_found}")

def verify_h5(directory, expected_dim):
    paths = list(Path(directory).glob('*.h5'))
    logger.info(f"Found {len(paths)} .h5 files.")
    
    corrupt = 0
    issues_found = 0
    for p in paths:
        try:
            with h5py.File(p, 'r') as f:
                for key in f.keys():
                    data = f[key][()]
                    issues = verify_tensor(data, expected_dim)
                    if issues:
                        logger.warning(f"Issues in {p.name} key {key}: {', '.join(issues)}")
                        issues_found += 1
        except Exception as e:
            logger.error(f"Corrupt H5 file {p.name}: {e}")
            corrupt += 1
            
    logger.info(f"H5 Verification complete. Corrupt: {corrupt}, Content Issues: {issues_found}")

def verify_lmdb(directory, expected_dim):
    paths = list(Path(directory).glob('*.lmdb'))
    logger.info(f"Found {len(paths)} .lmdb files.")
    
    corrupt = 0
    issues_found = 0
    for p in paths:
        try:
            env = lmdb.open(str(p), readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    data = np.frombuffer(value, dtype=np.float32)
                    issues = verify_tensor(data, expected_dim)
                    if issues:
                        logger.warning(f"Issues in {p.name} key {key.decode('utf-8')}: {', '.join(issues)}")
                        issues_found += 1
            env.close()
        except Exception as e:
            logger.error(f"Corrupt LMDB file {p.name}: {e}")
            corrupt += 1
            
    logger.info(f"LMDB Verification complete. Corrupt: {corrupt}, Content Issues: {issues_found}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--format", type=str, required=True, choices=["pt", "h5", "lmdb"])
    parser.add_argument("--expected_dim", type=int, default=None)
    args = parser.parse_args()

    if args.format == "pt":
        verify_pt(args.features_dir, args.expected_dim)
    elif args.format == "h5":
        verify_h5(args.features_dir, args.expected_dim)
    elif args.format == "lmdb":
        verify_lmdb(args.features_dir, args.expected_dim)

if __name__ == "__main__":
    main()
