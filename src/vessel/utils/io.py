"""Unified medical image I/O using SimpleITK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def load_image(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a medical image (NIfTI, MHA, NRRD, etc.) and return (array, metadata).

    Parameters
    ----------
    path : Path | str
        Path to the image file.

    Returns
    -------
    array : np.ndarray
        Image data as a NumPy array in (Z, Y, X) ordering for 3-D images.
    metadata : dict
        Dictionary with keys: spacing, origin, direction, size, dtype,
        pixel_id, and the original sitk_image for advanced use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(img)

    metadata: dict[str, Any] = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
        "dtype": array.dtype.name,
        "pixel_id": img.GetPixelIDTypeAsString(),
        "ndim": img.GetDimension(),
    }
    return array, metadata


def save_nifti(
    array: np.ndarray,
    metadata: dict[str, Any],
    path: Path | str,
    *,
    compress: bool = True,
) -> None:
    """Save a NumPy array as a NIfTI file (.nii.gz).

    Parameters
    ----------
    array : np.ndarray
        Image data (Z, Y, X) ordering.
    metadata : dict
        Must contain 'spacing', 'origin', 'direction'.
    path : Path | str
        Output file path.  Will be suffixed with .nii.gz if not already.
    compress : bool
        Whether to use gzip compression (default True).
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".nii.gz")

    path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(metadata.get("spacing", (1.0, 1.0, 1.0))))
    img.SetOrigin(tuple(metadata.get("origin", (0.0, 0.0, 0.0))))

    direction = metadata.get("direction")
    if direction is not None:
        img.SetDirection(tuple(direction))

    sitk.WriteImage(img, str(path), useCompression=compress)


def get_image_info(path: Path | str) -> dict[str, Any]:
    """Read image metadata without loading the full pixel array.

    Parameters
    ----------
    path : Path | str
        Path to the image file.

    Returns
    -------
    dict
        Keys: spacing, origin, direction, size, pixel_id, ndim, component_type.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    return {
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection(),
        "size": reader.GetSize(),
        "pixel_id": reader.GetPixelIDTypeAsString(),
        "ndim": reader.GetDimension(),
        "number_of_components": reader.GetNumberOfComponents(),
    }
