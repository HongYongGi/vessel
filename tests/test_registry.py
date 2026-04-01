"""Tests for the DatasetRegistry."""
from __future__ import annotations

import pytest


class TestRegistryListAll:
    """Tests for listing all datasets."""

    def test_registry_list_all(self, registry):
        """Registry should list all 7 registered datasets."""
        ids = registry.list_ids()
        assert len(ids) == 7
        assert "aortaseg24" in ids
        assert "topcow" in ids

    def test_registry_list_datasets_returns_configs(self, registry):
        """list_datasets() should return DatasetConfig objects."""
        datasets = registry.list_datasets()
        assert len(datasets) == 7
        for ds in datasets:
            assert ds.dataset.id
            assert ds.dataset.name


class TestRegistryFilterByTier:
    """Tests for filtering datasets by tier."""

    def test_registry_filter_by_tier(self, registry):
        """All current datasets are Tier 1."""
        tier1 = registry.list_datasets(tier=1)
        assert len(tier1) == 7

    def test_registry_filter_tier_no_results(self, registry):
        """Filtering by a non-existent tier should return an empty list."""
        tier99 = registry.list_datasets(tier=99)
        assert tier99 == []


class TestRegistryGetById:
    """Tests for getting a specific dataset by ID."""

    def test_registry_get_by_id(self, registry):
        """Should return the correct config for a known ID."""
        cfg = registry.get("aortaseg24")
        assert cfg.dataset.id == "aortaseg24"
        assert cfg.dataset.name == "AortaSeg24"

    def test_registry_get_nonexistent_raises(self, registry):
        """Getting a non-existent dataset should raise KeyError."""
        with pytest.raises(KeyError, match="not_a_real_dataset"):
            registry.get("not_a_real_dataset")


class TestRegistryListBodyRegions:
    """Tests for listing body regions."""

    def test_registry_list_body_regions(self, registry):
        """Should return a non-empty sorted list of unique body regions."""
        regions = registry.list_body_regions()
        assert isinstance(regions, list)
        assert len(regions) > 0
        # Should be sorted
        assert regions == sorted(regions)
        # No duplicates
        assert len(regions) == len(set(regions))

    def test_registry_filter_by_body_region(self, registry):
        """Filtering by body_region should return matching datasets."""
        # aortaseg24 has body_region=thoracoabdominal
        results = registry.list_datasets(body_region="thoraco")
        assert len(results) >= 1
        assert any(d.dataset.id == "aortaseg24" for d in results)
