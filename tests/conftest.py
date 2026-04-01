"""Shared test fixtures for vessel test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs/ directory."""
    return project_root / "configs"


@pytest.fixture
def sample_config(configs_dir: Path):
    """Load the aortaseg24 config as a representative sample."""
    from vessel.core.config import load_dataset_config

    return load_dataset_config(configs_dir / "aortaseg24.yaml")


@pytest.fixture
def all_configs(configs_dir: Path):
    """Load all dataset configs."""
    from vessel.core.config import load_all_configs

    return load_all_configs(configs_dir)


@pytest.fixture
def registry(configs_dir: Path):
    """Create a DatasetRegistry pointed at the real configs directory."""
    from vessel.core.registry import DatasetRegistry

    return DatasetRegistry(config_dir=configs_dir)
