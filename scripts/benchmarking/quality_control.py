import os
import json
import math

def check_quality(experiments_dir, expected_runs=5, output_report="quality_report.json"):
    """
    Performs quality control checks on benchmark runs to detect anomalies like
    missing runs, duplicate runs, and NaN values in metrics.
    """
    report = {
        "status": "passed",
        "missing_runs": {},
        "nan_detected": [],
        "duplicate_runs": []
    }
    
    if not os.path.exists(experiments_dir):
        report["status"] = "error"
        report["message"] = f"Directory '{experiments_dir}' not found."
        with open(output_report, "w") as f:
            json.dump(report, f, indent=4)
        return
        
    for exp_name in os.listdir(experiments_dir):
        exp_dir = os.path.join(experiments_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
            
        run_ids = []
        # Expect runs to be stored in subdirectories of the experiment
        for run_name in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_name)
            results_path = os.path.join(run_dir, "results.json")
            
            if os.path.isdir(run_dir) and os.path.exists(results_path):
                run_ids.append(run_name)
                
                try:
                    with open(results_path, "r") as f:
                        results = json.load(f)
                        
                    # Recursively check for NaNs in any numeric metric
                    def check_nans(data, prefix=""):
                        if isinstance(data, dict):
                            for k, v in data.items():
                                check_nans(v, prefix=f"{prefix}.{k}" if prefix else k)
                        elif isinstance(data, (int, float)) and math.isnan(data):
                            report["nan_detected"].append({
                                "experiment": exp_name,
                                "run": run_name,
                                "metric": prefix
                            })
                            report["status"] = "failed"
                    
                    check_nans(results)
                    
                except Exception as e:
                    print(f"Error reading {results_path}: {e}")
                    
        # Check for duplicate runs based on run_id
        seen = set()
        for rid in run_ids:
            if rid in seen:
                report["duplicate_runs"].append({
                    "experiment": exp_name,
                    "run": rid
                })
                report["status"] = "failed"
            seen.add(rid)
            
        # Check for missing runs
        if len(seen) < expected_runs:
            report["missing_runs"][exp_name] = {
                "expected": expected_runs,
                "found": len(seen)
            }
            report["status"] = "failed"
            
    with open(output_report, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Quality control finished. Status: {report['status']}")
    print(f"Detailed report saved to {output_report}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run quality control checks on benchmark results.")
    parser.add_argument("--experiments_dir", default="experiments", help="Directory containing experiment runs")
    parser.add_argument("--expected_runs", type=int, default=5, help="Expected number of runs/folds per experiment")
    parser.add_argument("--output", default="quality_report.json", help="Output JSON report file")
    args = parser.parse_args()
    
    check_quality(args.experiments_dir, args.expected_runs, args.output)
