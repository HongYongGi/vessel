"""Tests for dataset configuration loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestLoadSingleConfig:
    """Tests for loading a single dataset configuration."""

    def test_load_single_config(self, sample_config):
        """A single config file should parse into a valid DatasetConfig."""
        assert sample_config.dataset.id == "aortaseg24"
        assert sample_config.dataset.name == "AortaSeg24"
        assert sample_config.dataset.tier == 1
        assert sample_config.dataset.modality == "CTA"
        assert sample_config.labels.num_classes == 23
        assert 1 in sample_config.labels.mapping
        assert sample_config.labels.mapping[1] == "ascending_aorta"

    def test_config_has_source(self, sample_config):
        """Config should have a valid source section."""
        assert sample_config.source.type == "zenodo"

    def test_config_has_preprocess(self, sample_config):
        """Config should have preprocessing parameters."""
        assert sample_config.preprocess.target_spacing == [1.0, 1.0, 1.0]
        assert sample_config.preprocess.intensity_window.center == 300.0


class TestLoadAllConfigs:
    """Tests for loading all configurations from the configs directory."""

    def test_load_all_configs(self, all_configs):
        """Should load all 7 Tier 1 dataset configs."""
        assert len(all_configs) == 7
        expected_ids = {
            "aortaseg24",
            "imagecas",
            "mmwhs",
            "msd_hepatic_vessel",
            "parse2022",
            "totalsegmentator_v2",
            "topcow",
        }
        assert set(all_configs.keys()) == expected_ids

    def test_all_configs_have_required_fields(self, all_configs):
        """Every loaded config must have an id, name, and source type."""
        for dataset_id, cfg in all_configs.items():
            assert cfg.dataset.id == dataset_id
            assert cfg.dataset.name
            assert cfg.source.type


class TestConfigSchemaValidation:
    """Tests for schema validation on invalid input."""

    def test_invalid_config_missing_dataset(self, tmp_path):
        """A YAML missing the 'dataset' section should raise an error."""
        from vessel.core.config import load_dataset_config

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("version: '1.0'\nsource:\n  type: zenodo\n")

        with pytest.raises(Exception):
            load_dataset_config(bad_yaml)

    def test_invalid_config_bad_modality(self, tmp_path):
        """A config with an invalid modality value should raise a validation error."""
        from vessel.core.config import load_dataset_config

        content = (
            "version: '1.0'\n"
            "dataset:\n"
            "  id: test\n"
            "  name: Test\n"
            "  modality: INVALID\n"
            "source:\n"
            "  type: zenodo\n"
        )
        bad_yaml = tmp_path / "bad_modality.yaml"
        bad_yaml.write_text(content)

        with pytest.raises(Exception):
            load_dataset_config(bad_yaml)

    def test_nonexistent_config_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        from vessel.core.config import load_dataset_config

        with pytest.raises(FileNotFoundError):
            load_dataset_config(Path("/nonexistent/config.yaml"))
