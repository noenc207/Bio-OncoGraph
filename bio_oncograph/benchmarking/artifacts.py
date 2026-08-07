import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Any

class ReproducibilityPackage:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_git_diff(self) -> None:
        """Save the current git diff and commit hash."""
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
            diff = subprocess.check_output(['git', 'diff', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8')
            
            with open(os.path.join(self.output_dir, 'git_status.json'), 'w') as f:
                json.dump({'commit_hash': commit_hash}, f, indent=4)
                
            with open(os.path.join(self.output_dir, 'git_diff.patch'), 'w') as f:
                f.write(diff)
        except Exception:
            # Fallback if git is not available or not in a repo
            with open(os.path.join(self.output_dir, 'git_status.json'), 'w') as f:
                json.dump({'error': 'Could not extract git status.'}, f, indent=4)
            
    def save_env_json(self) -> None:
        """Save the current environment details."""
        try:
            pip_list = subprocess.check_output(['pip', 'list', '--format=json'], stderr=subprocess.STDOUT).decode('utf-8')
            packages = json.loads(pip_list)
            
            env_data = {
                'timestamp': datetime.now().isoformat(),
                'packages': packages
            }
            
            with open(os.path.join(self.output_dir, 'env.json'), 'w') as f:
                json.dump(env_data, f, indent=4)
        except Exception as e:
            with open(os.path.join(self.output_dir, 'env.json'), 'w') as f:
                json.dump({'error': str(e)}, f, indent=4)

    def save_dataset_manifest(self, datasets: Dict[str, str]) -> None:
        """
        Save the dataset manifest.
        datasets: Dict mapping dataset name to its file path or description.
        """
        manifest = {
            'datasets': datasets,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'dataset_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=4)
            
    def build(self, datasets: Dict[str, str]) -> None:
        """
        Build the complete reproducibility package.
        """
        self.save_git_diff()
        self.save_env_json()
        self.save_dataset_manifest(datasets)
