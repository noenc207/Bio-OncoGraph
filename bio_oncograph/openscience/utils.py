import datetime
import hashlib
import logging
import re

logger = logging.getLogger(__name__)

def sanitize_module_name(name: str) -> str:
    """Sanitize a string for use as a module or variable name.
    
    Args:
        name: The raw name string.
        
    Returns:
        A sanitized string safe for Python identifiers.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = sanitized.lstrip('0123456789_')
    if not sanitized:
        sanitized = "module"
    return sanitized.lower()

def generate_timestamp() -> str:
    """Generate an ISO-8601 formatted timestamp string.
    
    Returns:
        The current UTC timestamp string.
    """
    return datetime.datetime.utcnow().isoformat() + "Z"

def compute_content_hash(content: str) -> str:
    """Compute a SHA-256 hash of the given string content.
    
    Args:
        content: The input string.
        
    Returns:
        The hexadecimal digest of the hash.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def validate_semver(version: str) -> bool:
    """Validate if a string follows Semantic Versioning (SemVer) format.
    
    Args:
        version: The version string to check.
        
    Returns:
        True if valid SemVer, False otherwise.
    """
    pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-zA-Z0-9-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-zA-Z0-9-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    return bool(re.match(pattern, version))
