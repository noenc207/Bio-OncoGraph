import os
import json

def verify_fairness(experiments_dir, output_report="fairness_report.json"):
    """
    Verifies that all models in the benchmarking suite use identical hyperparameters
    where required for fair comparison.
    """
    models_data = {}
    
    if not os.path.exists(experiments_dir):
        print(f"Directory '{experiments_dir}' not found.")
        return
        
    for model_name in os.listdir(experiments_dir):
        model_dir = os.path.join(experiments_dir, model_name)
        config_path = os.path.join(model_dir, "config.json")
        
        if os.path.isdir(model_dir) and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                models_data[model_name] = config
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
                
    if not models_data:
        print("No models found with config.json.")
        with open(output_report, "w") as f:
            json.dump({"status": "no_data", "message": "No configurations found to compare."}, f, indent=4)
        return
        
    # Hyperparameters that must be strictly identical across all models for fairness
    fairness_keys = ["optimizer", "epochs", "batch_size", "patient_split", "seed"]
    
    reference_model = list(models_data.keys())[0]
    reference_config = models_data[reference_model]
    
    report = {
        "status": "passed",
        "mismatches": [],
        "reference_model": reference_model,
        "verified_keys": fairness_keys
    }
    
    for model_name, config in models_data.items():
        if model_name == reference_model:
            continue
            
        for key in fairness_keys:
            ref_val = reference_config.get(key)
            model_val = config.get(key)
            
            # Use string matching or strict equality
            if ref_val != model_val:
                report["status"] = "failed"
                report["mismatches"].append({
                    "model": model_name,
                    "key": key,
                    "expected": ref_val,
                    "actual": model_val
                })
                
    with open(output_report, "w") as f:
        json.dump(report, f, indent=4)
        
    if report["status"] == "passed":
        print(f"Fairness protocol check passed. Report saved to {output_report}")
    else:
        print(f"Fairness protocol check failed with {len(report['mismatches'])} mismatches. Report saved to {output_report}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify fairness protocol across benchmark models.")
    parser.add_argument("--experiments_dir", default="experiments", help="Directory containing experiment configs")
    parser.add_argument("--output", default="fairness_report.json", help="Output report JSON file")
    args = parser.parse_args()
    
    verify_fairness(args.experiments_dir, args.output)
