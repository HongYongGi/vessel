"""Pydantic models for dataset configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a single downloadable file."""

    name: str
    url: str | None = None
    sha256: str | None = None


class SourceConfig(BaseModel):
    """Configuration for dataset download source."""

    type: Literal["zenodo", "kaggle", "gdrive", "dropbox", "http", "grand_challenge"]
    # Zenodo
    zenodo_record_id: str | None = None
    record_id: str | None = None  # YAML alias for zenodo_record_id
    # Kaggle
    kaggle_dataset: str | None = None
    dataset: str | None = None  # YAML alias for kaggle_dataset
    # Google Drive
    gdrive_id: str | None = None
    file_id: str | None = None  # YAML alias for gdrive_id
    # HTTP direct
    url: str | None = None
    urls: list[str] = Field(default_factory=list)
    alt_url: str | None = None  # Alternative download URL
    # Grand Challenge
    challenge: str | None = None
    # Download filename
    filename: str | None = None
    # Files info
    files: list[FileInfo] = Field(default_factory=list)
    extract: Literal["zip", "tar", "tar.gz", "none"] = "none"

    def get_zenodo_record_id(self) -> str | None:
        """Return Zenodo record ID from either field name."""
        return self.zenodo_record_id or self.record_id

    def get_kaggle_dataset(self) -> str | None:
        """Return Kaggle dataset slug from either field name."""
        return self.kaggle_dataset or self.dataset

    def get_gdrive_id(self) -> str | None:
        """Return Google Drive file ID from either field name."""
        return self.gdrive_id or self.file_id


class FormatConfig(BaseModel):
    """Configuration for image/label file formats and patterns."""

    image_format: Literal["nifti", "mha", "dicom", "nrrd"] = "nifti"
    label_format: Literal["nifti", "mha", "dicom", "nrrd", "npz"] = "nifti"
    image_pattern: str = "**/*.nii.gz"
    label_pattern: str = "**/*.nii.gz"


class LabelConfig(BaseModel):
    """Configuration for label type and class mapping."""

    type: Literal["binary", "multiclass", "multi_file"] = "binary"
    num_classes: int = 1
    mapping: dict[int, str] = Field(default_factory=dict)
    file_mapping: dict[str, str] = Field(default_factory=dict)  # For multi_file label type


class IntensityWindow(BaseModel):
    """CT intensity windowing parameters."""

    center: float = 300.0
    width: float = 700.0


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    target_spacing: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity_window: IntensityWindow = Field(default_factory=IntensityWindow)
    crop_size: list[int] | None = None
    normalize: Literal["minmax", "zscore", "none"] = "minmax"


class DatasetInfo(BaseModel):
    """Core metadata about a dataset."""

    id: str
    name: str
    description: str = ""
    paper: str = ""
    license: str = ""
    tier: int = 1
    body_region: str = ""
    modality: Literal["CT", "CTA", "MRA", "MRI"] = "CT"
    num_cases: int = 0
    estimated_size_gb: float = 0.0


class DatasetConfig(BaseModel):
    """Top-level dataset configuration combining all sub-configs."""

    version: str = "1.0"
    dataset: DatasetInfo
    source: SourceConfig
    format: FormatConfig = Field(default_factory=FormatConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
