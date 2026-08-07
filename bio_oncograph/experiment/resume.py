"""Utilities for safely resuming studies without duplication."""
import logging
import optuna
from bio_oncograph.experiment.storage import get_storage

logger = logging.getLogger(__name__)

def resume_study(
    study_name: str,
    storage_uri: str,
) -> optuna.Study:
    """Safely resumes an existing study, locking it against conflicting setups.
    
    Args:
        study_name: The name of the study to resume.
        storage_uri: Database URI.
        
    Returns:
        The loaded Optuna study.
        
    Raises:
        ValueError: If the study does not exist in the given storage.
    """
    storage = get_storage(storage_uri)
    
    try:
        optuna.study.get_study_id(study_name=study_name, storage=storage)
    except KeyError as exc:
        logger.error(f"Study '{study_name}' does not exist in '{storage_uri}'.")
        raise ValueError(f"Cannot resume. Study '{study_name}' not found.") from exc
        
    logger.info(f"Successfully connected to existing study: {study_name}")
    
    return optuna.load_study(
        study_name=study_name,
        storage=storage
    )
