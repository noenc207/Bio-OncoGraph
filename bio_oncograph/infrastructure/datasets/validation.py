import os
import pandas as pd
from typing import List, Tuple

class DatasetValidator:
    """Validates dataset metadata and structure."""
    
    def __init__(self, metadata_path: str, wsi_dir: str):
        self.metadata_path = metadata_path
        self.wsi_dir = wsi_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def run_all_checks(self) -> Tuple[bool, List[str], List[str]]:
        """Runs comprehensive validation checks."""
        self.errors = []
        self.warnings = []
        
        if not os.path.exists(self.metadata_path):
            self.errors.append(f"Metadata file not found: {self.metadata_path}")
            return False, self.errors, self.warnings
            
        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            self.errors.append(f"Corrupted metadata file: {str(e)}")
            return False, self.errors, self.warnings

        self._check_duplicates(df)
        self._check_missing_wsi(df)
        self._check_missing_clinical(df)
        self._check_missing_survival(df)
        self._check_invalid_labels(df)
        
        return len(self.errors) == 0, self.errors, self.warnings

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        if 'patient_id' in df.columns:
            dups = df[df.duplicated('patient_id', keep=False)]
            if not dups.empty:
                self.warnings.append(f"Found {len(dups)} duplicate patient_ids.")
                
        if 'slide_id' in df.columns:
            dups = df[df.duplicated('slide_id', keep=False)]
            if not dups.empty:
                self.errors.append(f"Found {len(dups)} duplicate slide_ids.")
        else:
            self.errors.append("slide_id column is missing.")

    def _check_missing_wsi(self, df: pd.DataFrame) -> None:
        if 'slide_id' not in df.columns:
            return
            
        available_wsis = set(os.listdir(self.wsi_dir)) if os.path.exists(self.wsi_dir) else set()
        
        for slide_id in df['slide_id'].dropna():
            found = any(wsi.startswith(str(slide_id)) for wsi in available_wsis)
            if not found:
                self.errors.append(f"Missing WSI for slide_id: {slide_id}")

    def _check_missing_clinical(self, df: pd.DataFrame) -> None:
        clinical_cols = ['age', 'gender', 'stage']
        for col in clinical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    self.warnings.append(f"Missing {missing} values in clinical column '{col}'.")
            else:
                self.warnings.append(f"Clinical column '{col}' is missing.")

    def _check_missing_survival(self, df: pd.DataFrame) -> None:
        survival_cols = ['survival_time', 'event_occurred']
        for col in survival_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    self.errors.append(f"Missing {missing} values in survival column '{col}'.")
            else:
                self.errors.append(f"Survival column '{col}' is missing.")

    def _check_invalid_labels(self, df: pd.DataFrame) -> None:
        if 'label' in df.columns:
            missing = df['label'].isnull().sum()
            if missing > 0:
                self.errors.append(f"Missing {missing} values in 'label' column.")
