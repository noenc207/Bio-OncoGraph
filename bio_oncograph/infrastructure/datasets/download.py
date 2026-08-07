import os
import urllib.request
import urllib.error
import hashlib
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DownloadManager:
    """Manages downloading files with support for resumption, retries, mirrors, and checksum verification."""
    
    def __init__(self, mirrors: List[str], max_retries: int = 3, chunk_size: int = 8192):
        self.mirrors = mirrors
        self.max_retries = max_retries
        self.chunk_size = chunk_size

    def calculate_sha256(self, filepath: str) -> str:
        """Calculates the SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def download_file(self, filename: str, target_path: str, expected_sha256: Optional[str] = None) -> bool:
        """
        Downloads a file from the configured mirrors.
        Supports resuming partial downloads and verifies checksum if provided.
        """
        if os.path.exists(target_path):
            if expected_sha256:
                if self.calculate_sha256(target_path) == expected_sha256:
                    logger.info(f"File {target_path} already exists and checksum matches.")
                    return True
                else:
                    logger.warning(f"File {target_path} exists but checksum is invalid. Re-downloading.")
                    os.remove(target_path)
            else:
                logger.info(f"File {target_path} already exists. Skipping download.")
                return True

        for mirror in self.mirrors:
            url = f"{mirror.rstrip('/')}/{filename}"
            logger.info(f"Attempting to download from {url}")
            
            for attempt in range(self.max_retries):
                try:
                    headers = {}
                    if os.path.exists(target_path):
                        downloaded_size = os.path.getsize(target_path)
                        headers['Range'] = f'bytes={downloaded_size}-'
                    else:
                        downloaded_size = 0

                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        mode = 'ab' if downloaded_size > 0 else 'wb'
                        with open(target_path, mode) as f:
                            while True:
                                chunk = response.read(self.chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                    
                    if expected_sha256:
                        if self.calculate_sha256(target_path) != expected_sha256:
                            logger.error(f"Checksum mismatch for {target_path} after download.")
                            os.remove(target_path)
                            continue
                    
                    logger.info(f"Successfully downloaded {filename} to {target_path}")
                    return True
                    
                except urllib.error.HTTPError as e:
                    if e.code == 416: # Range not satisfiable
                        logger.info(f"File is already fully downloaded or invalid range.")
                        return True
                    logger.error(f"HTTP error {e.code} while downloading {url}: {e.reason}")
                except Exception as e:
                    logger.error(f"Error downloading {url} (Attempt {attempt + 1}/{self.max_retries}): {e}")
                
                time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"Failed to download {filename} after trying all mirrors.")
        return False
