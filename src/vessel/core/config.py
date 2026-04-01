"""YAML configuration loader for dataset configs."""
from __future__ import annotations

from pathlib import Path

import yaml

from vessel.core.metadata import DatasetConfig


def get_configs_dir() -> Path:
    """Return the default configs/ directory relative to the package root.

    The configs directory is located at the project root, two levels above
    the vessel package (src/vessel/../../configs).
    """
    package_dir = Path(__file__).resolve().parent.parent  # src/vessel
    project_root = package_dir.parent.parent  # project root
    return project_root / "configs"


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load a single dataset configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    DatasetConfig
        Parsed and validated dataset configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return DatasetConfig(**raw)


def load_all_configs(config_dir: Path | None = None) -> dict[str, DatasetConfig]:
    """Load all dataset configurations from a directory.

    Scans for ``*.yaml`` and ``*.yml`` files in the given directory.

    Parameters
    ----------
    config_dir : Path | None
        Directory containing YAML config files.  Defaults to :func:`get_configs_dir`.

    Returns
    -------
    dict[str, DatasetConfig]
        Mapping of dataset ID to its configuration.
    """
    if config_dir is None:
        config_dir = get_configs_dir()

    config_dir = Path(config_dir)
    configs: dict[str, DatasetConfig] = {}

    if not config_dir.exists():
        return configs

    for pattern in ("*.yaml", "*.yml"):
        for config_path in sorted(config_dir.glob(pattern)):
            # Skip non-dataset files (e.g. _label_taxonomy.yaml, _schema.yaml)
            if config_path.stem.startswith("_"):
                continue
            try:
                cfg = load_dataset_config(config_path)
                configs[cfg.dataset.id] = cfg
            except Exception as exc:
                # Log but don't crash on individual bad configs
                import warnings

                warnings.warn(
                    f"Failed to load config {config_path.name}: {exc}",
                    stacklevel=2,
                )

    return configs
