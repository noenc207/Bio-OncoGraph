import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)

class CommunicationBackend(ABC):
    """Abstract base class for communication backends."""
    
    @abstractmethod
    def send(self, recipient_id: str, message: Any) -> None:
        """Sends a message to a recipient.
        
        Args:
            recipient_id: The ID of the recipient.
            message: The payload to send.
        """
        pass
        
    @abstractmethod
    def receive(self, timeout: float = None) -> Any:
        """Receives a message.
        
        Args:
            timeout: Maximum time to wait for a message.
            
        Returns:
            The received message.
        """
        pass

class SyncBackend(CommunicationBackend):
    """Mock synchronous RPC communication backend."""
    
    def __init__(self) -> None:
        """Initializes SyncBackend."""
        logger.info("Initialized SyncBackend.")
        self.message_store: Dict[str, Any] = {}
        
    def send(self, recipient_id: str, message: Any) -> None:
        """Sends a message synchronously to the store."""
        logger.debug("Sync sending to %s", recipient_id)
        self.message_store[recipient_id] = message
        
    def receive(self, timeout: float = None) -> Any:
        """Retrieves a message synchronously."""
        logger.debug("Sync receive called.")
        return self.message_store.pop("global", None)

class AsyncBackend(CommunicationBackend):
    """Mock asynchronous message queue communication backend."""
    
    def __init__(self) -> None:
        """Initializes AsyncBackend."""
        logger.info("Initialized AsyncBackend.")
        self.queue: list = []
        
    def send(self, recipient_id: str, message: Any) -> None:
        """Queues a message asynchronously."""
        logger.debug("Async sending to %s", recipient_id)
        self.queue.append((recipient_id, message))
        
    def receive(self, timeout: float = None) -> Any:
        """Dequeues a message asynchronously."""
        start_time = time.time()
        while not self.queue:
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning("Receive timed out.")
                return None
            time.sleep(0.1)
        return self.queue.pop(0)

class RetryPolicy:
    """Simple backoff retry logic for simulated network failures."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        """Initializes RetryPolicy.
        
        Args:
            max_retries: Maximum number of retries.
            base_delay: Base delay for exponential backoff.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        logger.info("Initialized RetryPolicy with %d retries.", max_retries)
        
    def execute(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Executes a function with retry logic.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            The result of the function execution.
            
        Raises:
            Exception: If all retries fail.
        """
        attempt = 0
        while attempt <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Max retries reached. Failing.")
                    raise e
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning("Attempt %d failed: %s. Retrying in %f seconds.", attempt, e, delay)
                time.sleep(delay)
