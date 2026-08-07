"""Module for generating study summaries and paper tables."""
import pandas as pd
import os

class ReportGenerator:
    """Generates study reports and tables suitable for papers."""
    
    def __init__(self, data: pd.DataFrame):
        """Initializes report generator.
        
        Args:
            data: DataFrame containing the data to be reported.
        """
        self.data = data
        
    def generate_paper_table(self, output_dir: str) -> None:
        """Generates tables for LaTeX, Excel, and CSV formats.
        
        Args:
            output_dir: Directory to save the tables.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.data.to_csv(os.path.join(output_dir, "paper_table.csv"), index=False)
        self.data.to_excel(os.path.join(output_dir, "paper_table.xlsx"), index=False)
        
        with open(os.path.join(output_dir, "paper_table.tex"), 'w') as f:
            f.write(self.data.to_latex(index=False))
            
    def generate_study_summary(self, output_dir: str, summary_text: str) -> None:
        """Generates a markdown summary of the study.
        
        Args:
            output_dir: Directory to save the summary.
            summary_text: Content of the summary.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "study_summary.md"), 'w') as f:
            f.write(f"# Study Summary\n\n{summary_text}\n")
