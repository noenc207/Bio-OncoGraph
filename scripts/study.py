"""Hydra entrypoint for resuming/inspecting HPO studies."""
import logging
import os
import sys
import argparse

# Ensure module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bio_oncograph.experiment.resume import resume_study

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main execution entrypoint for resuming a study."""
    parser = argparse.ArgumentParser(description="Resume an Optuna study.")
    parser.add_argument("--study_name", type=str, required=True, help="Name of the study.")
    parser.add_argument("--storage_uri", type=str, required=True, help="Database URI.")
    
    args = parser.parse_args()
    
    study = resume_study(
        study_name=args.study_name,
        storage_uri=args.storage_uri
    )
    
    logger.info(f"Study '{study.study_name}' loaded.")
    if study.best_trial:
        logger.info(f"Best trial value: {study.best_trial.value}")
        logger.info(f"Best trial params: {study.best_trial.params}")

if __name__ == "__main__":
    main()
