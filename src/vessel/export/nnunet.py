"""nnUNet v2 format exporter for vessel segmentation datasets.

Output structure
----------------
nnUNet_raw/
└── Dataset{task_id}_{name}/
    ├── dataset.json
    ├── imagesTr/    # {case_id}_0000.nii.gz
    ├── labelsTr/    # {case_id}.nii.gz
    └── imagesTs/    # {case_id}_0000.nii.gz
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vessel.core.config import get_configs_dir
from vessel.core.metadata import DatasetConfig
from vessel.core.paths import get_export_dir, get_processed_dir
from vessel.core.registry import DatasetRegistry


def _load_label_taxonomy(configs_dir: Path | None = None) -> dict[str, Any]:
    """Load the unified label taxonomy from _label_taxonomy.yaml."""
    if configs_dir is None:
        configs_dir = get_configs_dir()
    taxonomy_path = configs_dir / "_label_taxonomy.yaml"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Label taxonomy not found: {taxonomy_path}")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NnUNetExporter:
    """Export processed datasets to nnUNet v2 raw format.

    Parameters
    ----------
    export_base : Path | None
        Root directory for nnUNet exports.
        Defaults to ``$VESSEL_DATA_ROOT/exports/nnUNet_raw``.
    registry : DatasetRegistry | None
        Pre-built registry instance.  A new one is created when *None*.
    """

    def __init__(
        self,
        export_base: Path | None = None,
        registry: DatasetRegistry | None = None,
    ) -> None:
        if export_base is None:
            export_base = get_export_dir() / "nnUNet_raw"
        self.export_base = Path(export_base)
        self.registry = registry or DatasetRegistry()
        self._taxonomy = _load_label_taxonomy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_single(
        self,
        dataset_id: str,
        task_id: int,
        task_name: str | None = None,
        label_subset: list[str] | None = None,
    ) -> Path:
        """Export a single dataset to nnUNet format.

        Parameters
        ----------
        dataset_id : str
            ID of the dataset to export (must exist in the registry).
        task_id : int
            nnUNet task/dataset ID (e.g. 500).
        task_name : str | None
            Human-readable name.  Defaults to the dataset's config name.
        label_subset : list[str] | None
            If given, only keep these label names; zero out all others.

        Returns
        -------
        Path
            Path to the created ``Dataset{task_id}_{name}`` directory.
        """
        cfg = self.registry.get(dataset_id)
        if task_name is None:
            task_name = cfg.dataset.name

        dataset_dir = self._make_dataset_dir(task_id, task_name)

        processed_dir = get_processed_dir(dataset_id)
        splits = self._load_splits(processed_dir)

        # Determine label mapping for export
        labels_for_export = self._resolve_labels(cfg, label_subset)

        train_ids = splits.get("train", [])
        test_ids = splits.get("test", [])

        images_tr = dataset_dir / "imagesTr"
        labels_tr = dataset_dir / "labelsTr"
        images_ts = dataset_dir / "imagesTs"
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)
        images_ts.mkdir(parents=True, exist_ok=True)

        # Copy training data
        for case_id in train_ids:
            self._copy_image(processed_dir, case_id, images_tr, case_id)
            self._copy_label(
                processed_dir, case_id, labels_tr, case_id,
                cfg=cfg, label_subset=label_subset, labels_for_export=labels_for_export,
            )

        # Copy test images
        for case_id in test_ids:
            self._copy_image(processed_dir, case_id, images_ts, case_id)

        # Generate dataset.json
        self._generate_dataset_json(
            dataset_dir=dataset_dir,
            task_name=task_name,
            labels=labels_for_export,
            num_training=len(train_ids),
            channel_names={"0": cfg.dataset.modality},
        )

        return dataset_dir

    def export_merged(
        self,
        dataset_ids: list[str],
        task_id: int,
        task_name: str,
        label_subset: list[str] | None = None,
    ) -> Path:
        """Merge multiple datasets into one nnUNet dataset.

        Parameters
        ----------
        dataset_ids : list[str]
            IDs of datasets to merge.
        task_id : int
            nnUNet task/dataset ID.
        task_name : str
            Human-readable name for the merged dataset.
        label_subset : list[str] | None
            If given, remap only these label names.

        Returns
        -------
        Path
            Path to the created dataset directory.
        """
        dataset_dir = self._make_dataset_dir(task_id, task_name)

        images_tr = dataset_dir / "imagesTr"
        labels_tr = dataset_dir / "labelsTr"
        images_ts = dataset_dir / "imagesTs"
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)
        images_ts.mkdir(parents=True, exist_ok=True)

        # Build unified label map from taxonomy
        unified_labels = self._build_unified_labels(dataset_ids, label_subset)

        total_train = 0
        modalities: set[str] = set()

        for dataset_id in dataset_ids:
            cfg = self.registry.get(dataset_id)
            modalities.add(cfg.dataset.modality)
            processed_dir = get_processed_dir(dataset_id)
            splits = self._load_splits(processed_dir)

            # Use a prefix derived from dataset id to avoid case_id conflicts
            prefix = self._dataset_prefix(dataset_id)

            train_ids = splits.get("train", [])
            test_ids = splits.get("test", [])

            for case_id in train_ids:
                out_case = f"{prefix}_{case_id}"
                self._copy_image(processed_dir, case_id, images_tr, out_case)
                self._copy_label_remapped(
                    processed_dir, case_id, labels_tr, out_case,
                    cfg=cfg, unified_labels=unified_labels,
                )
            total_train += len(train_ids)

            for case_id in test_ids:
                out_case = f"{prefix}_{case_id}"
                self._copy_image(processed_dir, case_id, images_ts, out_case)

        # Determine modality string
        modality_str = sorted(modalities)[0] if modalities else "CT"

        self._generate_dataset_json(
            dataset_dir=dataset_dir,
            task_name=task_name,
            labels=unified_labels,
            num_training=total_train,
            channel_names={"0": modality_str},
        )

        return dataset_dir

    # ------------------------------------------------------------------
    # dataset.json generation
    # ------------------------------------------------------------------

    def _generate_dataset_json(
        self,
        dataset_dir: Path,
        task_name: str,
        labels: dict[int, str],
        num_training: int,
        channel_names: dict[str, str] | None = None,
    ) -> Path:
        """Generate nnUNet v2 ``dataset.json`` file.

        Parameters
        ----------
        dataset_dir : Path
            Root directory of the nnUNet dataset.
        task_name : str
            Name of the dataset / task.
        labels : dict[int, str]
            ``{label_index: label_name}`` including 0 for background.
        num_training : int
            Number of training cases.
        channel_names : dict[str, str] | None
            Channel name mapping, e.g. ``{"0": "CT"}``.

        Returns
        -------
        Path
            Path to the written ``dataset.json``.
        """
        if channel_names is None:
            channel_names = {"0": "CT"}

        # nnUNet v2 expects labels as {name: index}
        labels_dict: dict[str, int] = {}
        for idx, name in sorted(labels.items()):
            labels_dict[name] = idx

        dataset_json = {
            "channel_names": channel_names,
            "labels": labels_dict,
            "numTraining": num_training,
            "file_ending": ".nii.gz",
        }

        out_path = dataset_dir / "dataset.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_dataset_dir(self, task_id: int, task_name: str) -> Path:
        """Create and return ``nnUNet_raw/Dataset{task_id}_{name}``."""
        safe_name = task_name.replace(" ", "_")
        dir_name = f"Dataset{task_id:03d}_{safe_name}"
        dataset_dir = self.export_base / dir_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    @staticmethod
    def _load_splits(processed_dir: Path) -> dict[str, list[str]]:
        """Load ``splits.json`` from the processed directory.

        If no splits file is found, treat all cases as training.
        """
        splits_path = processed_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback: detect case IDs from images/ directory
        images_dir = processed_dir / "images"
        if images_dir.exists():
            case_ids = sorted(
                p.name.replace(".nii.gz", "")
                for p in images_dir.glob("*.nii.gz")
            )
            return {"train": case_ids, "test": []}
        return {"train": [], "test": []}

    def _resolve_labels(
        self,
        cfg: DatasetConfig,
        label_subset: list[str] | None,
    ) -> dict[int, str]:
        """Build the label map for export.

        Returns a dict ``{new_index: name}`` including background at 0.
        If *label_subset* is given, only include those labels (re-indexed
        starting from 1).
        """
        original_mapping = cfg.labels.mapping  # {orig_idx: name}

        if label_subset is None:
            result: dict[int, str] = {0: "background"}
            for idx, name in sorted(original_mapping.items()):
                result[idx] = name
            return result

        # Re-index from 1 for the requested subset
        result = {0: "background"}
        new_idx = 1
        for _orig_idx, name in sorted(original_mapping.items()):
            if name in label_subset:
                result[new_idx] = name
                new_idx += 1
        return result

    def _build_unified_labels(
        self,
        dataset_ids: list[str],
        label_subset: list[str] | None,
    ) -> dict[int, str]:
        """Build a unified label map from the taxonomy for merging.

        Collects all label names that appear across the given datasets.
        If *label_subset* is specified, filters to only those names.
        Uses taxonomy IDs for consistent indexing.
        """
        taxonomy = self._taxonomy.get("taxonomy", {})

        # Collect all label names across datasets
        all_names: set[str] = set()
        for dataset_id in dataset_ids:
            cfg = self.registry.get(dataset_id)
            for name in cfg.labels.mapping.values():
                all_names.add(name)

        if label_subset is not None:
            all_names = all_names & set(label_subset)

        # Map to taxonomy IDs
        result: dict[int, str] = {0: "background"}
        for name in sorted(all_names):
            if name in taxonomy:
                tid = taxonomy[name]["id"]
                result[tid] = name
            else:
                # Fallback: assign a high index
                max_idx = max(result.keys()) if result else 0
                result[max_idx + 1] = name
        return result

    @staticmethod
    def _copy_image(
        processed_dir: Path,
        case_id: str,
        dest_dir: Path,
        out_case_id: str,
    ) -> None:
        """Copy a processed image to the nnUNet destination."""
        src = processed_dir / "images" / f"{case_id}.nii.gz"
        dst = dest_dir / f"{out_case_id}_0000.nii.gz"
        if src.exists():
            shutil.copy2(src, dst)

    def _copy_label(
        self,
        processed_dir: Path,
        case_id: str,
        dest_dir: Path,
        out_case_id: str,
        *,
        cfg: DatasetConfig,
        label_subset: list[str] | None,
        labels_for_export: dict[int, str],
    ) -> None:
        """Copy (and optionally filter) a label to the nnUNet destination."""
        src = processed_dir / "labels" / f"{case_id}.nii.gz"
        dst = dest_dir / f"{out_case_id}.nii.gz"

        if not src.exists():
            return

        if label_subset is None:
            # Direct copy — no remapping needed
            shutil.copy2(src, dst)
            return

        # Need to remap: zero out non-selected labels, re-index
        from vessel.utils.io import load_image, save_nifti

        array, metadata = load_image(src)
        new_array = np.zeros_like(array)

        # Build remap table: orig_idx -> new_idx
        remap: dict[int, int] = {}
        new_idx = 1
        for orig_idx, name in sorted(cfg.labels.mapping.items()):
            if name in label_subset:
                remap[orig_idx] = new_idx
                new_idx += 1

        for orig_idx, new_idx in remap.items():
            new_array[array == orig_idx] = new_idx

        save_nifti(new_array, metadata, dst)

    def _copy_label_remapped(
        self,
        processed_dir: Path,
        case_id: str,
        dest_dir: Path,
        out_case_id: str,
        *,
        cfg: DatasetConfig,
        unified_labels: dict[int, str],
    ) -> None:
        """Copy a label, remapping to the unified taxonomy indices."""
        src = processed_dir / "labels" / f"{case_id}.nii.gz"
        dst = dest_dir / f"{out_case_id}.nii.gz"

        if not src.exists():
            return

        from vessel.utils.io import load_image, save_nifti

        array, metadata = load_image(src)
        new_array = np.zeros_like(array)

        # Build remap: orig_idx -> unified_idx
        name_to_unified = {name: idx for idx, name in unified_labels.items()}

        for orig_idx, name in cfg.labels.mapping.items():
            if name in name_to_unified:
                new_array[array == orig_idx] = name_to_unified[name]

        save_nifti(new_array, metadata, dst)

    @staticmethod
    def _dataset_prefix(dataset_id: str) -> str:
        """Generate a short prefix from a dataset ID.

        Examples: ``aortaseg24`` -> ``aort``, ``topcow`` -> ``topc``
        """
        # Use first 4 characters as abbreviation
        return dataset_id[:4]
