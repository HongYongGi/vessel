"""Zenodo record downloader.

Downloads files from Zenodo using the REST API.  Supports automatic file
discovery from the record metadata, resume via HTTP Range header, and
SHA-256 integrity verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from vessel.core.metadata import SourceConfig
from vessel.download.base import BaseDownloader, DownloaderFactory
from vessel.utils.hash import verify_sha256
from vessel.utils.progress import vessel_progress

_ZENODO_API = "https://zenodo.org/api/records"
_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


class ZenodoDownloader(BaseDownloader):
    """Download files from a Zenodo record.

    If the source config does not list specific files, the downloader
    queries the Zenodo API and downloads every file attached to the record.
    """

    def check_available(self) -> bool:
        """Check whether the Zenodo record is reachable."""
        record_id = self.source.zenodo_record_id
        if not record_id:
            return False
        try:
            resp = requests.get(f"{_ZENODO_API}/{record_id}", timeout=15)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def download(self, resume: bool = True) -> list[Path]:
        """Download all files from the Zenodo record.

        Parameters
        ----------
        resume : bool
            If True, skip already-completed files and resume partial ones.

        Returns
        -------
        list[Path]
            Paths of downloaded files.
        """
        record_id = self.source.zenodo_record_id
        if not record_id:
            raise ValueError("zenodo_record_id is not set in source config")

        files_meta = self._discover_files(record_id)
        downloaded: list[Path] = []

        self._log_progress(
            f"Zenodo record {record_id}: {len(files_meta)} file(s) to download"
        )

        for idx, fmeta in enumerate(files_meta, 1):
            filename: str = fmeta["key"]
            url: str = fmeta["links"]["self"]
            remote_size: int = fmeta.get("size", 0)
            checksum: str | None = fmeta.get("checksum")  # "md5:..." or sha256

            dest_path = self.dest_dir / filename

            # Skip if already downloaded and verified
            if dest_path.exists() and dest_path.stat().st_size == remote_size:
                sha = self._find_sha256(filename)
                if sha is None or verify_sha256(dest_path, sha):
                    self._log_progress(
                        f"[{idx}/{len(files_meta)}] 이미 완료: {filename}"
                    )
                    downloaded.append(dest_path)
                    continue

            self._download_file(url, dest_path, remote_size, resume=resume, label=filename)

            # Verify integrity
            sha = self._find_sha256(filename)
            if sha and not verify_sha256(dest_path, sha):
                raise RuntimeError(
                    f"SHA-256 verification failed for {filename}. "
                    "The file may be corrupted — try downloading again."
                )

            downloaded.append(dest_path)

        return downloaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_files(self, record_id: str) -> list[dict[str, Any]]:
        """Fetch file metadata from the Zenodo API.

        If the source config already lists files with URLs, those are
        preferred.  Otherwise, every file in the record is returned.
        """
        resp = requests.get(f"{_ZENODO_API}/{record_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()

        all_files: list[dict[str, Any]] = data.get("files", [])
        if not all_files:
            raise RuntimeError(
                f"Zenodo record {record_id} has no files attached."
            )

        # If the config specifies particular filenames, filter down.
        if self.source.files:
            wanted = {fi.name for fi in self.source.files}
            filtered = [f for f in all_files if f["key"] in wanted]
            if filtered:
                return filtered

        return all_files

    def _find_sha256(self, filename: str) -> str | None:
        """Look up the SHA-256 hash for *filename* from the source config."""
        for fi in self.source.files:
            if fi.name == filename and fi.sha256:
                return fi.sha256
        return None

    def _download_file(
        self,
        url: str,
        dest: Path,
        total_size: int,
        *,
        resume: bool,
        label: str,
    ) -> None:
        """Stream-download a single file with progress bar and resume support."""
        headers: dict[str, str] = {}
        mode = "wb"
        initial = 0

        if resume and dest.exists():
            existing = dest.stat().st_size
            if existing < total_size:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                initial = existing
                self._log_progress(f"이어받기: {label} ({existing} bytes부터)")

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        effective_total = total_size if total_size else None

        pbar = vessel_progress(
            iterable=None,
            total=effective_total,
            desc=label,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        pbar.n = initial
        pbar.refresh()

        try:
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        finally:
            pbar.close()


# Register with the factory
DownloaderFactory.register("zenodo", ZenodoDownloader)
