"""Storage configuration and connection management for Optuna."""
import logging
from typing import Union
import optuna
from optuna.storages import RDBStorage, InMemoryStorage

logger = logging.getLogger(__name__)

def get_storage(storage_uri: str) -> Union[RDBStorage, InMemoryStorage]:
    """Instantiates the appropriate Optuna storage backend.
    
    Args:
        storage_uri: Database URI (e.g., 'sqlite:///example.db', 
                     'postgresql://user:pass@localhost/db', or 'memory').
                     
    Returns:
        An initialized Optuna storage object.
        
    Raises:
        ValueError: If the storage_uri format is unsupported.
    """
    if not storage_uri or storage_uri == "memory":
        logger.info("Using InMemoryStorage for Optuna.")
        return InMemoryStorage()
        
    if storage_uri.startswith("sqlite") or storage_uri.startswith("postgresql"):
        logger.info(f"Using RDBStorage for Optuna with URI: {storage_uri.split('@')[-1]}")
        return RDBStorage(
            url=storage_uri,
            engine_kwargs={"pool_pre_ping": True}
        )
        
    raise ValueError(f"Unsupported storage URI scheme: {storage_uri}")
