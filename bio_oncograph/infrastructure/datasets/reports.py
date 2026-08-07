"""
Dataset Reporter for Bio-OncoGraph.
Generates statistics and reports for datasets.
"""
import csv
import json
import os
from collections import Counter
from typing import Any, Dict, Sequence

class DatasetReporter:
    """Generates robust reports and statistics for datasets."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_all(self, samples: Sequence[Dict[str, Any]]) -> None:
        """Generates all reports from the samples."""
        self.generate_dataset_statistics(samples)
        self.generate_class_distribution(samples)
        self.generate_patient_distribution(samples)
        self.generate_survival_statistics(samples)
        self.generate_missing_values(samples)
        self.generate_markdown_report(samples)
        
    def generate_dataset_statistics(self, samples: Sequence[Dict[str, Any]]) -> None:
        stats = {
            "total_samples": len(samples),
            "total_patients": len(set(s.get("patient_id") for s in samples if s.get("patient_id"))),
            "modalities": list(set(s.get("modality") for s in samples if s.get("modality"))),
        }
        with open(os.path.join(self.output_dir, "dataset_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
    def generate_class_distribution(self, samples: Sequence[Dict[str, Any]]) -> None:
        labels = [s.get("target") for s in samples if s.get("target") is not None]
        counts = Counter(labels)
        
        with open(os.path.join(self.output_dir, "class_distribution.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Count"])
            for cls_val, count in sorted(counts.items()):
                writer.writerow([cls_val, count])
                
    def generate_patient_distribution(self, samples: Sequence[Dict[str, Any]]) -> None:
        patient_counts = Counter(s.get("patient_id") for s in samples if s.get("patient_id"))
        
        with open(os.path.join(self.output_dir, "patient_distribution.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PatientID", "SampleCount"])
            for pid, count in sorted(patient_counts.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([pid, count])
                
    def generate_survival_statistics(self, samples: Sequence[Dict[str, Any]]) -> None:
        survival_data = [
            (s.get("survival_time"), s.get("event")) 
            for s in samples if s.get("survival_time") is not None and s.get("event") is not None
        ]
        
        with open(os.path.join(self.output_dir, "survival_statistics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SurvivalTime", "Event"])
            for time, event in survival_data:
                writer.writerow([time, event])
                
    def generate_missing_values(self, samples: Sequence[Dict[str, Any]]) -> None:
        keys = set()
        for s in samples:
            keys.update(s.keys())
            
        missing_counts = {k: 0 for k in keys}
        for s in samples:
            for k in keys:
                if s.get(k) is None:
                    missing_counts[k] += 1
                    
        with open(os.path.join(self.output_dir, "missing_values.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "MissingCount"])
            for field, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([field, count])
                
    def generate_markdown_report(self, samples: Sequence[Dict[str, Any]]) -> None:
        total_samples = len(samples)
        total_patients = len(set(s.get("patient_id") for s in samples if s.get("patient_id")))
        
        report = [
            "# Dataset Report",
            "",
            "## Summary",
            f"- **Total Samples:** {total_samples}",
            f"- **Total Patients:** {total_patients}",
            ""
        ]
        
        with open(os.path.join(self.output_dir, "dataset_report.md"), "w") as f:
            f.write("\n".join(report))
