"""Execution entrypoint for comparing models or running ablation studies."""
import argparse
import sys
from bio_oncograph.experiment.ablation import AblationStudy
from bio_oncograph.experiment.comparison import ModelComparison

def main():
    """Main execution function for compare script."""
    parser = argparse.ArgumentParser(description="Compare models or run ablation studies.")
    parser.add_argument("--mode", type=str, choices=["ablation", "comparison"], required=True, 
                        help="Mode of operation: 'ablation' or 'comparison'.")
    args = parser.parse_args()
    
    if args.mode == "ablation":
        print("Running ablation study...")
        # Note: In a real scenario, data loading logic would be here.
        # For demonstration purposes, we print the intended action.
    elif args.mode == "comparison":
        print("Running model comparison...")
        # Note: In a real scenario, data loading logic would be here.
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
