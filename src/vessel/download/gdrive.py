"""Google Drive downloader.

Uses the ``gdown`` library to download files from Google Drive, including
large files that trigger a virus-scan confirmation page.
"""

from __future__ import annotations

from pathlib import Path

from vessel.core.metadata import SourceConfig
from vessel.download.base import BaseDownloader, DownloaderFactory
from vessel.utils.hash import verify_sha256


class GDriveDownloader(BaseDownloader):
    """Download files from Google Drive using gdown."""

    def check_available(self) -> bool:
        """Check whether gdown is installed and a drive ID is configured."""
        try:
            import gdown  # noqa: F401
        except ImportError:
            return False

        return bool(self.source.get_gdrive_id())

    def download(self, resume: bool = True) -> list[Path]:
        """Download a file from Google Drive.

        Parameters
        ----------
        resume : bool
            If True, gdown will attempt to resume partial downloads.

        Returns
        -------
        list[Path]
            Paths of downloaded files.
        """
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                "gdown is required for Google Drive downloads.\n"
                "Install: pip install gdown"
            )

        gdrive_id = self.source.get_gdrive_id()
        if not gdrive_id:
            raise ValueError("gdrive_id (or file_id) is not set in source config")

        url = f"https://drive.google.com/uc?id={gdrive_id}"

        # Determine output filename
        if self.source.files:
            output_name = self.source.files[0].name
        else:
            output_name = None  # gdown will auto-detect

        self._log_progress(f"Google Drive 다운로드 시작: {gdrive_id}")

        output_path = str(self.dest_dir / output_name) if output_name else str(self.dest_dir) + "/"

        result = gdown.download(
            url,
            output=output_path,
            quiet=False,
            resume=resume,
            fuzzy=True,  # handles various GDrive URL formats
        )

        if result is None:
            raise RuntimeError(
                f"Google Drive download failed for file ID: {gdrive_id}\n"
                "The file may be private or require access permission."
            )

        dest_path = Path(result)

        # Verify integrity if SHA-256 is available
        if self.source.files:
            sha = self.source.files[0].sha256
            if sha and not verify_sha256(dest_path, sha):
                raise RuntimeError(
                    f"SHA-256 verification failed for {dest_path.name}. "
                    "The file may be corrupted — try downloading again."
                )

        self._log_progress(f"다운로드 완료: {dest_path.name}")
        return [dest_path]


# Register with the factory
DownloaderFactory.register("gdrive", GDriveDownloader)
