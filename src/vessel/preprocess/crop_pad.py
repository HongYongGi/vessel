"""Crop and pad utilities for image-label pairs."""

from __future__ import annotations

import numpy as np


def crop_to_foreground(
    image_array: np.ndarray,
    label_array: np.ndarray,
    margin: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and label to the non-zero bounding box of the label.

    A *margin* (in voxels) is added on every side.  The crop is clamped so
    it never exceeds the original array bounds.

    Parameters
    ----------
    image_array : np.ndarray
        Input image (Z, Y, X) or (C, Z, Y, X).
    label_array : np.ndarray
        Label map with the same spatial dimensions as *image_array*.
    margin : int
        Number of voxels to pad around the bounding box on each side.

    Returns
    -------
    cropped_image : np.ndarray
    cropped_label : np.ndarray
    """
    # Find non-zero voxel coordinates in the label
    nonzero = np.argwhere(label_array > 0)
    if nonzero.size == 0:
        # No foreground -- return as-is
        return image_array, label_array

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)

    # Build slice objects with margin clamped to array shape
    slices = tuple(
        slice(max(0, mn - margin), min(s, mx + margin + 1))
        for mn, mx, s in zip(mins, maxs, label_array.shape)
    )

    return image_array[slices], label_array[slices]


def pad_to_size(
    array: np.ndarray,
    target_size: list[int],
    pad_value: float = 0,
) -> np.ndarray:
    """Centre-pad *array* so its spatial dimensions reach *target_size*.

    If the array is already larger than *target_size* along a given axis
    that axis is left unchanged (no cropping).

    Parameters
    ----------
    array : np.ndarray
        Input array of any dimensionality.
    target_size : list[int]
        Desired size for each dimension.  Must have the same length as
        ``array.ndim``.
    pad_value : float
        Constant value used for padding.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    if len(target_size) != array.ndim:
        raise ValueError(
            f"target_size length ({len(target_size)}) must match "
            f"array ndim ({array.ndim})."
        )

    pad_widths: list[tuple[int, int]] = []
    for current, target in zip(array.shape, target_size):
        if current >= target:
            pad_widths.append((0, 0))
        else:
            total_pad = target - current
            before = total_pad // 2
            after = total_pad - before
            pad_widths.append((before, after))

    if all(p == (0, 0) for p in pad_widths):
        return array

    return np.pad(array, pad_widths, mode="constant", constant_values=pad_value)
