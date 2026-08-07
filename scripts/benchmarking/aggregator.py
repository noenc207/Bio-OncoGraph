import os
import json
import csv
import glob
from collections import defaultdict
import statistics
import argparse

def aggregate_metrics(input_dir=".", output_dir="."):
    search_pattern = os.path.join(input_dir, "**", "*.json")
    
    data_by_model = defaultdict(lambda: defaultdict(list))
    
    for filepath in glob.glob(search_pattern, recursive=True):
        if os.path.basename(filepath) in ['summary.json']:
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if "model" not in data or "seed" not in data:
                continue
                
            model = data["model"]
            
            for key, value in data.items():
                if key in ["model", "seed"]:
                    continue
                if isinstance(value, (int, float)):
                    data_by_model[model][key].append(value)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    summary_data = {}
    
    for model, metrics in data_by_model.items():
        summary_data[model] = {}
        for metric_name, values in metrics.items():
            if len(values) == 0:
                continue
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            
            summary_data[model][f"{metric_name}_mean"] = mean_val
            summary_data[model][f"{metric_name}_std"] = std_val
            summary_data[model]["samples"] = len(values)
            
    os.makedirs(output_dir, exist_ok=True)
    json_out = os.path.join(output_dir, "summary.json")
    with open(json_out, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    csv_out = os.path.join(output_dir, "summary.csv")
    if summary_data:
        all_columns = set()
        for model, stats in summary_data.items():
            all_columns.update(stats.keys())
            
        columns = ["model"] + sorted(list(all_columns))
        
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for model, stats in summary_data.items():
                row = {"model": model}
                row.update(stats)
                writer.writerow(row)
                
    print(f"Aggregated data saved to {json_out} and {csv_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate benchmark metrics.")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory to search for json metrics.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save summaries.")
    args = parser.parse_args()
    
    aggregate_metrics(args.input_dir, args.output_dir)
