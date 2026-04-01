"""Intensity windowing and normalisation for CT images."""

from __future__ import annotations

import numpy as np


def apply_intensity_window(
    image_array: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """Apply CT Hounsfield-unit windowing and normalise to [0, 1].

    The window is defined as ``[center - width/2, center + width/2]``.
    Values outside this range are clipped, then the result is linearly
    rescaled to [0, 1].

    Parameters
    ----------
    image_array : np.ndarray
        Input image in HU (or any intensity scale).
    center : float
        Window centre level.
    width : float
        Window width.

    Returns
    -------
    np.ndarray
        Windowed array with values in [0, 1], dtype float32.
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    arr = np.clip(image_array, lower, upper).astype(np.float32)
    # Avoid division by zero when width == 0
    if width > 0:
        arr = (arr - lower) / (upper - lower)
    else:
        arr = np.zeros_like(arr)
    return arr


def normalize(
    image_array: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalise an image array.

    Parameters
    ----------
    image_array : np.ndarray
        Input array.
    method : str
        ``"minmax"`` -> rescale to [0, 1].
        ``"zscore"`` -> zero mean, unit variance.
        ``"none"``   -> return a float32 copy unchanged.

    Returns
    -------
    np.ndarray
        Normalised array, dtype float32.

    Raises
    ------
    ValueError
        If *method* is not one of the recognised options.
    """
    arr = image_array.astype(np.float32)

    if method == "minmax":
        mn, mx = arr.min(), arr.max()
        if mx - mn > 0:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
    elif method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
    elif method == "none":
        pass
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose from: 'minmax', 'zscore', 'none'."
        )

    return arr
