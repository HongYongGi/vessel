"""Path management for vessel dataset directories.

All paths are derived from the VESSEL_DATA_ROOT environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "VESSEL_DATA_ROOT"


def _validate_dataset_id(dataset_id: str) -> str:
    """Validate dataset_id to prevent path traversal.

    Raises
    ------
    ValueError
        If the dataset_id contains unsafe characters.
    """
    if not dataset_id or ".." in dataset_id or "/" in dataset_id or "\\" in dataset_id:
        raise ValueError(
            f"Invalid dataset_id: {dataset_id!r}. "
            "Must not be empty or contain '..', '/', or '\\'."
        )
    return dataset_id


def get_data_root() -> Path:
    """Return the data root directory from VESSEL_DATA_ROOT env var.

    Raises
    ------
    EnvironmentError
        If VESSEL_DATA_ROOT is not set or is empty.
    """
    root = os.environ.get(_ENV_VAR)
    if not root or not root.strip():
        raise EnvironmentError(
            f"Environment variable '{_ENV_VAR}' is not set. "
            f"Please set it to the desired data directory, e.g.:\n"
            f"  export {_ENV_VAR}=/path/to/vessel_data"
        )
    return Path(root)


def get_raw_dir(dataset_id: str) -> Path:
    """Return the raw data directory for a specific dataset."""
    return get_data_root() / "raw" / _validate_dataset_id(dataset_id)


def get_processed_dir(dataset_id: str) -> Path:
    """Return the processed data directory for a specific dataset."""
    return get_data_root() / "processed" / _validate_dataset_id(dataset_id)


def get_export_dir() -> Path:
    """Return the exports directory."""
    return get_data_root() / "exports"


def get_status_dir() -> Path:
    """Return the internal status directory."""
    return get_data_root() / ".vessel"


def ensure_dirs(dataset_id: str) -> None:
    """Create all required directories for a dataset.

    Creates raw, processed, exports, and status directories.
    """
    dirs = [
        get_raw_dir(dataset_id),
        get_processed_dir(dataset_id),
        get_export_dir(),
        get_status_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
