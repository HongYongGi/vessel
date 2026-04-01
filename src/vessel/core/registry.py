"""Dataset registry for discovering and accessing dataset configurations."""
from __future__ import annotations

from pathlib import Path

from vessel.core.config import get_configs_dir, load_all_configs
from vessel.core.metadata import DatasetConfig


class DatasetRegistry:
    """Central registry that manages all known dataset configurations.

    Parameters
    ----------
    config_dir : Path | None
        Directory containing YAML config files.  When *None*, the package's
        built-in ``configs/`` directory is used.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            config_dir = get_configs_dir()
        self._configs: dict[str, DatasetConfig] = load_all_configs(config_dir)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_datasets(
        self,
        tier: int | None = None,
        body_region: str | None = None,
    ) -> list[DatasetConfig]:
        """Return dataset configs, optionally filtered by tier and/or body region.

        Parameters
        ----------
        tier : int | None
            If given, only return datasets with this tier value.
        body_region : str | None
            If given, only return datasets whose body_region matches
            (case-insensitive substring match).
        """
        results: list[DatasetConfig] = []
        for cfg in self._configs.values():
            if tier is not None and cfg.dataset.tier != tier:
                continue
            if body_region is not None and body_region.lower() not in cfg.dataset.body_region.lower():
                continue
            results.append(cfg)
        return results

    def get(self, dataset_id: str) -> DatasetConfig:
        """Return the configuration for a specific dataset.

        Raises
        ------
        KeyError
            If the dataset ID is not found in the registry.
        """
        try:
            return self._configs[dataset_id]
        except KeyError:
            available = ", ".join(sorted(self._configs.keys())) or "(none)"
            raise KeyError(
                f"Dataset '{dataset_id}' not found. Available: {available}"
            ) from None

    def list_ids(self) -> list[str]:
        """Return a sorted list of all registered dataset IDs."""
        return sorted(self._configs.keys())

    def list_body_regions(self) -> list[str]:
        """Return a sorted list of unique body regions across all datasets."""
        regions: set[str] = set()
        for cfg in self._configs.values():
            region = cfg.dataset.body_region.strip()
            if region:
                regions.add(region)
        return sorted(regions)

    def get_label_mapping(self, dataset_id: str) -> dict[int, str]:
        """Return the label mapping for a specific dataset.

        Parameters
        ----------
        dataset_id : str
            The dataset identifier.

        Returns
        -------
        dict[int, str]
            Mapping of label index to label name.
        """
        cfg = self.get(dataset_id)
        return cfg.labels.mapping
