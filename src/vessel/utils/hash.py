"""File integrity utilities — SHA-256 hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Read in 8 MiB chunks to keep memory usage low for large files.
_CHUNK_SIZE = 8 * 1024 * 1024


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file.

    Parameters
    ----------
    file_path : Path | str
        Path to the file.

    Returns
    -------
    str
        Lowercase hex-encoded SHA-256 hash string.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def verify_sha256(file_path: Path | str, expected: str) -> bool:
    """Verify that a file matches the expected SHA-256 hash.

    Parameters
    ----------
    file_path : Path | str
        Path to the file.
    expected : str
        Expected hex-encoded SHA-256 hash (case-insensitive).

    Returns
    -------
    bool
        ``True`` if the computed hash matches *expected*.
    """
    actual = compute_sha256(file_path)
    return actual.lower() == expected.strip().lower()
