"""Module for generating leaderboards."""
import pandas as pd
import json
import os
from typing import List, Dict, Any

class Leaderboard:
    """Manages leaderboard generation and serialization."""
    
    def __init__(self):
        """Initializes leaderboard."""
        self.entries: List[Dict[str, Any]] = []
        
    def add_entry(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Adds an entry to the leaderboard.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of metric names to values.
        """
        entry = {"model": model_name}
        entry.update(metrics)
        self.entries.append(entry)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Converts entries to DataFrame.
        
        Returns:
            Pandas DataFrame containing leaderboard entries.
        """
        return pd.DataFrame(self.entries)
        
    def save(self, output_dir: str) -> None:
        """Saves leaderboard to CSV, Markdown, and JSON formats.
        
        Args:
            output_dir: Directory to save the leaderboard files.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self.to_dataframe()
        
        df.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)
        df.to_markdown(os.path.join(output_dir, "leaderboard.md"), index=False)
        
        with open(os.path.join(output_dir, "leaderboard.json"), 'w') as f:
            json.dump(self.entries, f, indent=4)
