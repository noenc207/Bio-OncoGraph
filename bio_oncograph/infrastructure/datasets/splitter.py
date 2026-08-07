"""
Deterministic Splitter for Bio-OncoGraph.
Supports Patient-level splitting (no leakage), Stratified splits, and K-Fold.
"""
import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Sequence

logger = logging.getLogger("BioOncoGraph")

class DeterministicSplitter:
    """
    Deterministic Splitter ensuring strict patient-level separation.
    """
    
    @staticmethod
    def _group_by_patient(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        patients = defaultdict(list)
        for sample in samples:
            patient_id = sample.get("patient_id")
            if not patient_id:
                raise ValueError("Sample is missing 'patient_id'")
            patients[patient_id].append(sample)
        return dict(patients)
        
    @staticmethod
    def _get_patient_label(patient_samples: List[Dict[str, Any]]) -> Any:
        # Assume the label is in "target" or "label"
        labels = []
        for s in patient_samples:
            if "target" in s:
                labels.append(s["target"])
            elif "label" in s:
                labels.append(s["label"])
        
        if not labels:
            return None
            
        # Check if consistent
        first = labels[0]
        if all(l == first for l in labels):
            return first
            
        # Return most frequent
        return max(set(labels), key=labels.count)

    @classmethod
    def split(
        cls,
        samples: Sequence[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratify: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
            
        rng = random.Random(seed)
        patients = cls._group_by_patient(samples)
        
        if stratify:
            patient_labels = {pid: cls._get_patient_label(samps) for pid, samps in patients.items()}
            label_to_pids = defaultdict(list)
            for pid, label in patient_labels.items():
                label_to_pids[label].append(pid)
                
            train_pids, val_pids, test_pids = [], [], []
            
            for label, pids in label_to_pids.items():
                rng.shuffle(pids)
                n = len(pids)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_pids.extend(pids[:n_train])
                val_pids.extend(pids[n_train:n_train + n_val])
                test_pids.extend(pids[n_train + n_val:])
        else:
            pids = list(patients.keys())
            rng.shuffle(pids)
            n = len(pids)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_pids = pids[:n_train]
            val_pids = pids[n_train:n_train + n_val]
            test_pids = pids[n_train + n_val:]
            
        return {
            "train": [s for pid in train_pids for s in patients[pid]],
            "val": [s for pid in val_pids for s in patients[pid]],
            "test": [s for pid in test_pids for s in patients[pid]]
        }
        
    @classmethod
    def k_fold(
        cls,
        samples: Sequence[Dict[str, Any]],
        k: int = 5,
        seed: int = 42,
        stratify: bool = False
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        if k < 2:
            raise ValueError("k must be >= 2")
            
        rng = random.Random(seed)
        patients = cls._group_by_patient(samples)
        
        buckets = [[] for _ in range(k)]
        
        if stratify:
            patient_labels = {pid: cls._get_patient_label(samps) for pid, samps in patients.items()}
            label_to_pids = defaultdict(list)
            for pid, label in patient_labels.items():
                label_to_pids[label].append(pid)
                
            for label, pids in label_to_pids.items():
                rng.shuffle(pids)
                for i, pid in enumerate(pids):
                    buckets[i % k].append(pid)
        else:
            pids = list(patients.keys())
            rng.shuffle(pids)
            for i, pid in enumerate(pids):
                buckets[i % k].append(pid)
                
        folds = []
        for fold_idx in range(k):
            val_pids = buckets[fold_idx]
            train_pids = []
            for j in range(k):
                if j != fold_idx:
                    train_pids.extend(buckets[j])
                    
            folds.append({
                "train": [s for pid in train_pids for s in patients[pid]],
                "val": [s for pid in val_pids for s in patients[pid]]
            })
            
        return folds

    @staticmethod
    def export_manifest(split_result: Dict[str, List[Dict[str, Any]]], filepath: str) -> None:
        manifest = {}
        for split_name, split_samples in split_result.items():
            manifest[split_name] = [s.get("sample_id") for s in split_samples if "sample_id" in s]
            
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    @staticmethod
    def import_manifest(filepath: str, all_samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        with open(filepath, 'r') as f:
            manifest = json.load(f)
            
        sample_map = {s.get("sample_id"): s for s in all_samples if "sample_id" in s}
        
        result = {}
        for split_name, sample_ids in manifest.items():
            result[split_name] = [sample_map[sid] for sid in sample_ids if sid in sample_map]
            
        return result
