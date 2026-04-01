"""Kaggle dataset downloader.

Downloads datasets using the ``kaggle`` CLI tool.  Requires the ``kaggle``
package to be installed and API credentials to be configured
(``~/.kaggle/kaggle.json``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from vessel.core.metadata import SourceConfig
from vessel.download.base import BaseDownloader, DownloaderFactory


class KaggleDownloader(BaseDownloader):
    """Download datasets from Kaggle via the kaggle CLI."""

    def check_available(self) -> bool:
        """Check whether the kaggle CLI is installed and credentials exist."""
        # Check CLI availability
        if shutil.which("kaggle") is None:
            return False

        # Check credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            return False

        # Verify the dataset identifier is configured
        if not self.source.kaggle_dataset:
            return False

        return True

    def download(self, resume: bool = True) -> list[Path]:
        """Download the Kaggle dataset.

        Parameters
        ----------
        resume : bool
            Not directly supported by kaggle CLI — if files already exist
            they will be re-downloaded (kaggle CLI behavior).

        Returns
        -------
        list[Path]
            Paths of downloaded files (typically a single zip).
        """
        dataset = self.source.kaggle_dataset
        if not dataset:
            raise ValueError("kaggle_dataset is not set in source config")

        if not self.check_available():
            raise RuntimeError(
                "Kaggle CLI is not available or credentials are missing.\n"
                "Install: pip install kaggle\n"
                "Then place your API key at ~/.kaggle/kaggle.json\n"
                "See: https://github.com/Kaggle/kaggle-api#api-credentials"
            )

        self._log_progress(f"Kaggle 데이터셋 다운로드 시작: {dataset}")

        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(self.dest_dir),
        ]

        # --unzip is not used here; extraction is handled by extract.py
        self._log_progress(f"실행: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed (exit code {result.returncode}):\n"
                f"{result.stderr}"
            )

        if result.stdout:
            self._log_progress(result.stdout.strip())

        # Collect downloaded files
        downloaded: list[Path] = []
        for p in self.dest_dir.iterdir():
            if p.is_file():
                downloaded.append(p)

        self._log_progress(f"다운로드 완료: {len(downloaded)} 파일")
        return downloaded


# Register with the factory
DownloaderFactory.register("kaggle", KaggleDownloader)
