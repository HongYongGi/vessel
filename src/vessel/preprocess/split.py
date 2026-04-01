"""Train / validation / test split generation for datasets."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_splits(
    case_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate reproducible train / val / test splits.

    Parameters
    ----------
    case_ids : list[str]
        List of case identifiers to split.
    train_ratio, val_ratio, test_ratio : float
        Proportions for each split.  Must sum to 1.0 (within tolerance).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, list[str]]
        ``{"train": [...], "val": [...], "test": [...]}``.

    Raises
    ------
    ValueError
        If ratios do not sum to ~1.0 or case list is empty.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})."
        )
    if not case_ids:
        raise ValueError("case_ids list is empty.")

    ids = sorted(case_ids)  # sort for determinism
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Remainder goes to test to avoid off-by-one
    n_test = n - n_train - n_val

    splits = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }

    logger.info(
        "Generated splits: train=%d, val=%d, test=%d (total=%d, seed=%d)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        n,
        seed,
    )
    return splits


def save_splits(splits: dict[str, list[str]], output_path: Path) -> None:
    """Save splits dictionary to a JSON file.

    Parameters
    ----------
    splits : dict[str, list[str]]
        Split mapping as returned by :func:`generate_splits`.
    output_path : Path
        Destination JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2, ensure_ascii=False)

    logger.info("Splits saved to %s", output_path)


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load splits from a JSON file.

    Parameters
    ----------
    splits_path : Path
        Path to the JSON file created by :func:`save_splits`.

    Returns
    -------
    dict[str, list[str]]
        Split mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
