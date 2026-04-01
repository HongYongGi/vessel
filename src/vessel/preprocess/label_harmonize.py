"""Harmonise dataset-specific labels to a unified taxonomy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default taxonomy path: configs/_label_taxonomy.yaml shipped with the package
_DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "_label_taxonomy.yaml"
)


def load_taxonomy(taxonomy_path: Path | None = None) -> dict:
    """Load the ``_label_taxonomy.yaml`` file.

    Parameters
    ----------
    taxonomy_path : Path | None
        Path to a custom taxonomy YAML.  When ``None`` the package default
        (``configs/_label_taxonomy.yaml``) is used.

    Returns
    -------
    dict
        Parsed YAML content with keys ``taxonomy`` and ``dataset_mappings``.

    Raises
    ------
    FileNotFoundError
        If the taxonomy file does not exist.
    """
    path = taxonomy_path or _DEFAULT_TAXONOMY_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


def get_unified_mapping(dataset_id: str, taxonomy: dict) -> dict[int, int]:
    """Build a mapping from dataset-local label IDs to unified taxonomy IDs.

    The function looks up the dataset in ``taxonomy["dataset_mappings"]``,
    then resolves each anatomy name to its numeric ``id`` in
    ``taxonomy["taxonomy"]``.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier key in ``dataset_mappings``.
    taxonomy : dict
        Full taxonomy dict as returned by :func:`load_taxonomy`.

    Returns
    -------
    dict[int, int]
        Mapping ``{original_label_value: unified_taxonomy_id}``.
        For example ``{1: 80, 2: 104}`` means dataset label 1 maps to
        unified ID 80 (``hepatic_vessel``).

    Raises
    ------
    KeyError
        If the dataset or an anatomy name is not found in the taxonomy.
    """
    dataset_map = taxonomy.get("dataset_mappings", {})
    if dataset_id not in dataset_map:
        available = ", ".join(sorted(dataset_map.keys())) or "(none)"
        raise KeyError(
            f"Dataset '{dataset_id}' not found in taxonomy dataset_mappings. "
            f"Available: {available}"
        )

    tax = taxonomy.get("taxonomy", {})
    mapping: dict[int, int] = {}

    for orig_label, anatomy_name in dataset_map[dataset_id].items():
        if anatomy_name not in tax:
            raise KeyError(
                f"Anatomy name '{anatomy_name}' (from dataset '{dataset_id}', "
                f"label {orig_label}) not found in taxonomy."
            )
        unified_id = tax[anatomy_name]["id"]
        mapping[int(orig_label)] = int(unified_id)

    return mapping


def harmonize_labels(
    label_array: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    """Remap label values using a precomputed mapping.

    Uses a NumPy lookup table for O(1) per-voxel mapping, which is much
    faster than iterating over unique values for large arrays.

    Parameters
    ----------
    label_array : np.ndarray
        Integer label array (any shape).
    mapping : dict[int, int]
        ``{old_value: new_value}`` pairs.

    Returns
    -------
    np.ndarray
        Remapped label array with the same shape, dtype ``int32``.
    """
    if not mapping:
        return label_array.astype(np.int32)

    max_old = max(int(label_array.max()), max(mapping.keys()))
    lut = np.zeros(max_old + 1, dtype=np.int32)

    for old_val, new_val in mapping.items():
        if old_val <= max_old:
            lut[old_val] = new_val

    # Ensure the input can index into the LUT
    safe_arr = np.clip(label_array.astype(np.int64), 0, max_old)
    return lut[safe_arr]
