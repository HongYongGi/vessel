"""Main preprocessing pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from vessel.core.metadata import DatasetConfig
from vessel.core.paths import get_raw_dir, get_processed_dir, ensure_dirs
from vessel.preprocess.resample import resample_image
from vessel.preprocess.intensity import apply_intensity_window, normalize
from vessel.preprocess.label_harmonize import (
    load_taxonomy,
    get_unified_mapping,
    harmonize_labels,
)
from vessel.preprocess.validate import validate_dataset
from vessel.preprocess.split import generate_splits, save_splits
from vessel.utils.io import save_nifti
from vessel.utils.progress import vessel_progress

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """Orchestrates the full preprocessing pipeline for a single dataset.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The parsed dataset configuration.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.config = dataset_config
        self.dataset_id = dataset_config.dataset.id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, workers: int = 1) -> dict[str, Any]:
        """Run the complete preprocessing pipeline.

        Steps:
            1. Discover raw image-label pairs.
            2. Resample + intensity window + label harmonise + save.
            3. Validate processed outputs.
            4. Generate train/val/test splits.

        Parameters
        ----------
        workers : int
            Number of parallel worker processes.  ``1`` means sequential.

        Returns
        -------
        dict
            Processing report with keys: ``total``, ``success``, ``failed``,
            ``errors``, ``validation``, ``splits_path``, ``elapsed_sec``.
        """
        t0 = time.time()
        ensure_dirs(self.dataset_id)

        raw_dir = get_raw_dir(self.dataset_id)
        output_dir = get_processed_dir(self.dataset_id)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # 1. Discover pairs
        pairs = self._discover_cases(raw_dir)
        if not pairs:
            logger.warning("No image-label pairs found in %s", raw_dir)
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "errors": {},
                "validation": {},
                "splits_path": None,
                "elapsed_sec": time.time() - t0,
            }

        logger.info("Discovered %d cases for %s", len(pairs), self.dataset_id)

        # 2. Process cases
        report: dict[str, Any] = {
            "total": len(pairs),
            "success": 0,
            "failed": 0,
            "errors": {},
        }

        # Build label mapping once (may raise KeyError if dataset not in taxonomy)
        label_mapping: dict[int, int] | None = None
        try:
            taxonomy = load_taxonomy()
            label_mapping = get_unified_mapping(self.dataset_id, taxonomy)
            logger.info(
                "Label mapping for %s: %s", self.dataset_id, label_mapping
            )
        except KeyError:
            logger.info(
                "No taxonomy mapping for %s, labels will be kept as-is.",
                self.dataset_id,
            )

        if workers <= 1:
            # Sequential processing with progress bar
            for img_path, lbl_path in vessel_progress(
                pairs, desc=f"전처리 중: {self.dataset_id}"
            ):
                result = self._process_single(
                    img_path, lbl_path, output_dir, label_mapping
                )
                if result.get("error"):
                    report["failed"] += 1
                    report["errors"][result["case_id"]] = result["error"]
                else:
                    report["success"] += 1
        else:
            # Parallel processing
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for img_path, lbl_path in pairs:
                    future = executor.submit(
                        _process_single_worker,
                        img_path,
                        lbl_path,
                        output_dir,
                        self.config.preprocess.target_spacing,
                        self.config.preprocess.intensity_window.center,
                        self.config.preprocess.intensity_window.width,
                        self.config.preprocess.normalize,
                        label_mapping,
                    )
                    futures[future] = (img_path, lbl_path)

                for future in vessel_progress(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"전처리 중: {self.dataset_id}",
                ):
                    result = future.result()
                    if result.get("error"):
                        report["failed"] += 1
                        report["errors"][result["case_id"]] = result["error"]
                    else:
                        report["success"] += 1

        # 3. Validate
        logger.info("Validating processed dataset ...")
        validation = validate_dataset(output_dir)
        report["validation"] = validation

        n_issues = sum(1 for v in validation.values() if v)
        if n_issues:
            logger.warning("%d cases have validation issues.", n_issues)
        else:
            logger.info("All cases passed validation.")

        # 4. Generate splits
        case_ids = sorted(
            f.name.replace(".nii.gz", "")
            for f in (output_dir / "images").glob("*.nii.gz")
        )
        if case_ids:
            splits = generate_splits(case_ids)
            splits_path = output_dir / "splits.json"
            save_splits(splits, splits_path)
            report["splits_path"] = str(splits_path)
        else:
            report["splits_path"] = None

        report["elapsed_sec"] = round(time.time() - t0, 2)
        logger.info(
            "Pipeline complete: %d/%d success, %d failed (%.1fs)",
            report["success"],
            report["total"],
            report["failed"],
            report["elapsed_sec"],
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_cases(self, raw_dir: Path) -> list[tuple[Path, Path]]:
        """Find image-label pairs using glob patterns from config.

        Returns
        -------
        list[tuple[Path, Path]]
            Sorted list of ``(image_path, label_path)`` tuples.
        """
        img_pattern = self.config.format.image_pattern
        lbl_pattern = self.config.format.label_pattern

        image_files = sorted(raw_dir.glob(img_pattern))
        label_files = sorted(raw_dir.glob(lbl_pattern))

        if not image_files:
            logger.warning("No images matching '%s' in %s", img_pattern, raw_dir)
            return []
        if not label_files:
            logger.warning("No labels matching '%s' in %s", lbl_pattern, raw_dir)
            return []

        # Build stem -> path maps for matching
        img_map: dict[str, Path] = {}
        for p in image_files:
            stem = p.name.replace(".nii.gz", "").replace(".nii", "")
            img_map[stem] = p

        lbl_map: dict[str, Path] = {}
        for p in label_files:
            stem = p.name.replace(".nii.gz", "").replace(".nii", "")
            lbl_map[stem] = p

        # Match by stem
        common_stems = sorted(set(img_map.keys()) & set(lbl_map.keys()))
        if not common_stems:
            # Fallback: pair by sorted order if stems don't match
            logger.warning(
                "No matching stems between images and labels. "
                "Falling back to positional pairing."
            )
            n = min(len(image_files), len(label_files))
            return list(zip(image_files[:n], label_files[:n]))

        return [(img_map[s], lbl_map[s]) for s in common_stems]

    def _process_single(
        self,
        image_path: Path,
        label_path: Path,
        output_dir: Path,
        label_mapping: dict[int, int] | None = None,
    ) -> dict[str, Any]:
        """Process one image-label pair.

        Returns
        -------
        dict
            ``{"case_id": str, "error": str | None}``.
        """
        case_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")
        try:
            # Load
            img_sitk = sitk.ReadImage(str(image_path))
            lbl_sitk = sitk.ReadImage(str(label_path))

            # Resample
            target_sp = self.config.preprocess.target_spacing
            img_sitk = resample_image(img_sitk, target_sp, is_label=False)
            lbl_sitk = resample_image(lbl_sitk, target_sp, is_label=True)

            # Convert to arrays
            img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
            lbl_arr = sitk.GetArrayFromImage(lbl_sitk)

            # Intensity windowing
            window = self.config.preprocess.intensity_window
            img_arr = apply_intensity_window(img_arr, window.center, window.width)

            # Normalisation (after windowing, already in [0,1] so mainly for zscore)
            norm_method = self.config.preprocess.normalize
            if norm_method != "none":
                img_arr = normalize(img_arr, method=norm_method) if norm_method != "minmax" else img_arr

            # Harmonise labels
            if label_mapping:
                lbl_arr = harmonize_labels(lbl_arr, label_mapping)

            # Build metadata for saving
            metadata = {
                "spacing": img_sitk.GetSpacing(),
                "origin": img_sitk.GetOrigin(),
                "direction": img_sitk.GetDirection(),
            }

            # Save
            save_nifti(img_arr, metadata, output_dir / "images" / f"{case_id}.nii.gz")
            save_nifti(
                lbl_arr.astype(np.int32),
                metadata,
                output_dir / "labels" / f"{case_id}.nii.gz",
            )

            return {"case_id": case_id, "error": None}

        except Exception as exc:
            logger.error("Failed to process %s: %s", case_id, exc)
            return {"case_id": case_id, "error": str(exc)}


# ------------------------------------------------------------------
# Module-level function for ProcessPoolExecutor (must be picklable)
# ------------------------------------------------------------------


def _process_single_worker(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    target_spacing: list[float],
    window_center: float,
    window_width: float,
    norm_method: str,
    label_mapping: dict[int, int] | None,
) -> dict[str, Any]:
    """Standalone worker function for parallel preprocessing.

    Mirrors :meth:`PreprocessPipeline._process_single` but receives all
    parameters explicitly so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    case_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")
    try:
        img_sitk = sitk.ReadImage(str(image_path))
        lbl_sitk = sitk.ReadImage(str(label_path))

        img_sitk = resample_image(img_sitk, target_spacing, is_label=False)
        lbl_sitk = resample_image(lbl_sitk, target_spacing, is_label=True)

        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk)

        img_arr = apply_intensity_window(img_arr, window_center, window_width)

        if norm_method not in ("none", "minmax"):
            img_arr = normalize(img_arr, method=norm_method)

        if label_mapping:
            lbl_arr = harmonize_labels(lbl_arr, label_mapping)

        metadata = {
            "spacing": img_sitk.GetSpacing(),
            "origin": img_sitk.GetOrigin(),
            "direction": img_sitk.GetDirection(),
        }

        save_nifti(img_arr, metadata, output_dir / "images" / f"{case_id}.nii.gz")
        save_nifti(
            lbl_arr.astype(np.int32),
            metadata,
            output_dir / "labels" / f"{case_id}.nii.gz",
        )

        return {"case_id": case_id, "error": None}

    except Exception as exc:
        return {"case_id": case_id, "error": str(exc)}
