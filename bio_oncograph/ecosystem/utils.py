import hashlib
import logging

logger = logging.getLogger(__name__)

def format_metric(value: float, precision: int) -> str:
    """Formats a metric value to a specific precision.
    
    Args:
        value: The numerical metric value.
        precision: Number of decimal places.
        
    Returns:
        Formatted string representation of the metric.
    """
    return f"{value:.{precision}f}"

def generate_experiment_id(name: str, seed: int) -> str:
    """Generates a unique experiment ID.
    
    Args:
        name: Name of the experiment.
        seed: Random seed used.
        
    Returns:
        A unique string identifier.
    """
    raw_id = f"{name}_{seed}"
    return hashlib.md5(raw_id.encode('utf-8')).hexdigest()[:10]

def compute_hash(content: str) -> str:
    """Computes a SHA-256 hash of the given content.
    
    Args:
        content: String content to hash.
        
    Returns:
        The hex digest of the hash.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string (e.g., '1h 5m 10s').
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divides two numbers, returning 0.0 if denominator is 0.
    
    Args:
        numerator: The numerator.
        denominator: The denominator.
        
    Returns:
        The result of the division, or 0.0.
    """
    if denominator == 0:
        logger.warning("Attempted division by zero")
        return 0.0
    return numerator / denominator
