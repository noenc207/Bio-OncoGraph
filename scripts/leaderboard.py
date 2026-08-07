"""Execution entrypoint for generating leaderboards and reports."""
import argparse
import sys
from bio_oncograph.experiment.leaderboard import Leaderboard
from bio_oncograph.experiment.report import ReportGenerator

def main():
    """Main execution function for leaderboard script."""
    parser = argparse.ArgumentParser(description="Generate leaderboard and reports.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the outputs.")
    args = parser.parse_args()
    
    # Initialize leaderboard
    board = Leaderboard()
    
    # Example data (in production, this would be loaded from experiment results)
    board.add_entry("Baseline", {"accuracy": 0.85, "f1_score": 0.84})
    board.add_entry("Model_V1", {"accuracy": 0.92, "f1_score": 0.91})
    board.add_entry("Model_V2", {"accuracy": 0.95, "f1_score": 0.94})
    
    # Save leaderboard formats
    board.save(args.output_dir)
    
    # Generate reports
    report = ReportGenerator(board.to_dataframe())
    report.generate_paper_table(args.output_dir)
    report.generate_study_summary(
        args.output_dir, 
        "This is an automated summary of the experiment performance across different models."
    )
    
    print(f"Generated leaderboard and reports in '{args.output_dir}'")
    return 0

if __name__ == "__main__":
    sys.exit(main())
