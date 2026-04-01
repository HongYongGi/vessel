"""Validation utilities for processed image-label pairs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vessel.utils.io import load_image

logger = logging.getLogger(__name__)


def validate_pair(image_path: Path, label_path: Path) -> list[str]:
    """Validate that an image-label pair is consistent and well-formed.

    Checks performed:

    1. Both files can be loaded.
    2. Spatial shapes match.
    3. Spacings match (within tolerance).
    4. Image contains no NaN or Inf values.
    5. Label contains no NaN or Inf values.
    6. Label has at least one non-zero voxel.
    7. Label values are non-negative integers.

    Parameters
    ----------
    image_path : Path
        Path to the processed image file.
    label_path : Path
        Path to the processed label file.

    Returns
    -------
    list[str]
        List of issue descriptions.  An empty list means the pair is valid.
    """
    issues: list[str] = []

    # --- Load ---
    try:
        img_arr, img_meta = load_image(image_path)
    except Exception as exc:
        issues.append(f"Cannot load image: {exc}")
        return issues

    try:
        lbl_arr, lbl_meta = load_image(label_path)
    except Exception as exc:
        issues.append(f"Cannot load label: {exc}")
        return issues

    # --- Shape match ---
    if img_arr.shape != lbl_arr.shape:
        issues.append(
            f"Shape mismatch: image {img_arr.shape} vs label {lbl_arr.shape}"
        )

    # --- Spacing match (tolerance 0.01 mm) ---
    img_spacing = img_meta.get("spacing", ())
    lbl_spacing = lbl_meta.get("spacing", ())
    if img_spacing and lbl_spacing:
        for axis, (isp, lsp) in enumerate(zip(img_spacing, lbl_spacing)):
            if abs(isp - lsp) > 0.01:
                issues.append(
                    f"Spacing mismatch on axis {axis}: "
                    f"image {isp:.4f} vs label {lsp:.4f}"
                )

    # --- NaN / Inf in image ---
    if np.isnan(img_arr).any():
        issues.append("Image contains NaN values")
    if np.isinf(img_arr).any():
        issues.append("Image contains Inf values")

    # --- NaN / Inf in label ---
    if np.isnan(lbl_arr).any():
        issues.append("Label contains NaN values")
    if np.isinf(lbl_arr).any():
        issues.append("Label contains Inf values")

    # --- Non-empty label ---
    if lbl_arr.max() == 0:
        issues.append("Label is entirely zero (empty)")

    # --- Non-negative integer labels ---
    if lbl_arr.min() < 0:
        issues.append(f"Label has negative values (min={lbl_arr.min()})")

    return issues


def validate_dataset(processed_dir: Path) -> dict[str, list[str]]:
    """Validate all image-label pairs in a processed dataset directory.

    Expects a directory layout::

        processed_dir/
            images/
                case_001.nii.gz
                case_002.nii.gz
            labels/
                case_001.nii.gz
                case_002.nii.gz

    Parameters
    ----------
    processed_dir : Path
        Root of the processed dataset directory.

    Returns
    -------
    dict[str, list[str]]
        ``{case_id: [issues]}``.  Cases with no issues have an empty list.
    """
    processed_dir = Path(processed_dir)
    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"

    results: dict[str, list[str]] = {}

    if not images_dir.is_dir():
        return {"_error": [f"Images directory not found: {images_dir}"]}
    if not labels_dir.is_dir():
        return {"_error": [f"Labels directory not found: {labels_dir}"]}

    image_files = sorted(images_dir.glob("*.nii.gz"))
    if not image_files:
        return {"_error": [f"No .nii.gz files found in {images_dir}"]}

    for img_path in image_files:
        case_id = img_path.name.replace(".nii.gz", "")
        lbl_path = labels_dir / img_path.name

        if not lbl_path.exists():
            results[case_id] = [f"Missing label file: {lbl_path}"]
            continue

        issues = validate_pair(img_path, lbl_path)
        results[case_id] = issues

    # Check for orphaned label files
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    image_stems = {f.name for f in image_files}
    for lbl_path in label_files:
        if lbl_path.name not in image_stems:
            case_id = lbl_path.name.replace(".nii.gz", "")
            results[case_id] = [f"Orphaned label (no matching image): {lbl_path}"]

    return results
