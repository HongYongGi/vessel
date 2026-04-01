"""Convert DICOM series to NIfTI format using SimpleITK."""

from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path) -> Path:
    """Convert a DICOM series directory to NIfTI using SimpleITK.

    If the directory contains multiple DICOM series, each series is converted
    separately with a ``_series{N}`` suffix.  When only a single series is
    present the output is written directly to *output_path*.

    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files (may include sub-directories).
    output_path : Path
        Desired output path for the NIfTI file (e.g. ``case_001.nii.gz``).
        For multi-series directories a suffix is appended before the extension.

    Returns
    -------
    Path
        Path to the (first) generated ``.nii.gz`` file.

    Raises
    ------
    FileNotFoundError
        If *dicom_dir* does not exist or contains no DICOM files.
    """
    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)

    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Discover all series IDs in the directory
    reader = sitk.ImageSeriesReader()
    series_ids: tuple[str, ...] = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for idx, series_id in enumerate(series_ids):
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if not dicom_names:
            logger.warning("Series %s has no files, skipping.", series_id)
            continue

        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        # Build the output filename
        if len(series_ids) == 1:
            dest = output_path
        else:
            stem = output_path.name.replace(".nii.gz", "").replace(".nii", "")
            dest = output_path.parent / f"{stem}_series{idx}.nii.gz"

        # Ensure .nii.gz extension
        if not str(dest).endswith(".nii.gz"):
            dest = dest.with_suffix(".nii.gz")

        sitk.WriteImage(image, str(dest), useCompression=True)
        logger.info(
            "Converted series %d/%d (%d slices) -> %s",
            idx + 1,
            len(series_ids),
            len(dicom_names),
            dest,
        )
        saved_paths.append(dest)

    return saved_paths[0]
