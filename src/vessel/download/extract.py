"""Archive extraction utilities.

Supports ZIP, TAR, TAR.GZ (and TGZ) archives with progress reporting.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from vessel.utils.progress import vessel_progress


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    format: str = "auto",
) -> Path:
    """Extract an archive to the destination directory.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file.
    dest_dir : Path
        Directory to extract into.
    format : str
        Archive format: ``"zip"``, ``"tar"``, ``"tar.gz"``, or ``"auto"``
        (detect from file extension).

    Returns
    -------
    Path
        The directory where files were extracted (``dest_dir``).

    Raises
    ------
    ValueError
        If the format cannot be determined or is unsupported.
    FileNotFoundError
        If the archive file does not exist.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        format = _detect_format(archive_path)

    if format == "zip":
        _extract_zip(archive_path, dest_dir)
    elif format in ("tar", "tar.gz", "tgz"):
        _extract_tar(archive_path, dest_dir, format)
    else:
        raise ValueError(
            f"Unsupported archive format: '{format}'. "
            "Supported: zip, tar, tar.gz"
        )

    return dest_dir


def _detect_format(path: Path) -> str:
    """Detect archive format from file extension."""
    name = path.name.lower()

    if name.endswith(".zip"):
        return "zip"
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    elif name.endswith(".tar"):
        return "tar"
    else:
        raise ValueError(
            f"Cannot determine archive format from filename: {path.name}. "
            "Specify the format explicitly."
        )


def _extract_zip(archive_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive with progress bar."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in vessel_progress(members, desc=f"압축 해제: {archive_path.name}"):
            zf.extract(member, dest_dir)


def _extract_tar(archive_path: Path, dest_dir: Path, format: str) -> None:
    """Extract a TAR (or TAR.GZ) archive with progress bar."""
    mode = "r:gz" if format in ("tar.gz", "tgz") else "r:"

    with tarfile.open(archive_path, mode) as tf:
        members = tf.getmembers()

        # Security: filter out absolute paths and path traversal
        safe_members = [
            m for m in members
            if not m.name.startswith("/") and ".." not in m.name
        ]

        for member in vessel_progress(safe_members, desc=f"압축 해제: {archive_path.name}"):
            tf.extract(member, dest_dir, filter="data")
