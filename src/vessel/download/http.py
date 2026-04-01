"""Direct HTTP/HTTPS downloader.

Downloads files from direct URLs with progress bars and resume support
via the HTTP Range header.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests

from vessel.core.metadata import SourceConfig
from vessel.download.base import BaseDownloader, DownloaderFactory
from vessel.utils.hash import verify_sha256
from vessel.utils.progress import vessel_progress

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def _filename_from_url(url: str) -> str:
    """Extract a filename from a URL path."""
    path = urlsplit(url).path
    name = unquote(path.rsplit("/", 1)[-1]) if "/" in path else "download"
    return name or "download"


class HTTPDownloader(BaseDownloader):
    """Download files from direct HTTP/HTTPS URLs."""

    def check_available(self) -> bool:
        """Check whether at least one URL is configured and reachable."""
        urls = self._collect_urls()
        if not urls:
            return False

        # Quick HEAD check on the first URL
        try:
            resp = requests.head(urls[0], timeout=15, allow_redirects=True)
            return resp.status_code < 400
        except requests.RequestException:
            return False

    def download(self, resume: bool = True) -> list[Path]:
        """Download all configured URLs.

        Parameters
        ----------
        resume : bool
            If True, resume partially downloaded files via Range header.

        Returns
        -------
        list[Path]
            Paths of downloaded files.
        """
        urls = self._collect_urls()
        if not urls:
            raise ValueError(
                "No URL configured. Set 'url' or 'urls' in the source config."
            )

        downloaded: list[Path] = []

        for idx, url in enumerate(urls, 1):
            # Determine filename from config or URL
            filename = self._get_filename(idx - 1, url)
            dest_path = self.dest_dir / filename

            # Get remote file size
            remote_size = self._get_remote_size(url)

            # Skip if already downloaded
            if dest_path.exists() and remote_size and dest_path.stat().st_size == remote_size:
                sha = self._find_sha256(idx - 1)
                if sha is None or verify_sha256(dest_path, sha):
                    self._log_progress(
                        f"[{idx}/{len(urls)}] 이미 완료: {filename}"
                    )
                    downloaded.append(dest_path)
                    continue

            self._log_progress(f"[{idx}/{len(urls)}] 다운로드 시작: {filename}")
            self._download_file(url, dest_path, remote_size, resume=resume, label=filename)

            # Verify integrity
            sha = self._find_sha256(idx - 1)
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

    def _collect_urls(self) -> list[str]:
        """Gather all URLs from the source config."""
        urls: list[str] = []
        if self.source.url:
            urls.append(self.source.url)
        urls.extend(self.source.urls)
        # Also include URLs from individual file entries
        for fi in self.source.files:
            if fi.url and fi.url not in urls:
                urls.append(fi.url)
        return urls

    def _get_filename(self, index: int, url: str) -> str:
        """Determine output filename for the given file index."""
        if index < len(self.source.files) and self.source.files[index].name:
            return self.source.files[index].name
        return _filename_from_url(url)

    def _find_sha256(self, index: int) -> str | None:
        """Look up the SHA-256 hash by file index."""
        if index < len(self.source.files):
            return self.source.files[index].sha256
        return None

    def _get_remote_size(self, url: str) -> int | None:
        """Issue a HEAD request to determine Content-Length."""
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
        except (requests.RequestException, ValueError):
            return None

    def _download_file(
        self,
        url: str,
        dest: Path,
        total_size: int | None,
        *,
        resume: bool,
        label: str,
    ) -> None:
        """Stream-download a file with progress bar and resume support."""
        headers: dict[str, str] = {}
        mode = "wb"
        initial = 0

        if resume and dest.exists() and total_size:
            existing = dest.stat().st_size
            if existing < total_size:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                initial = existing
                self._log_progress(f"이어받기: {label} ({existing} bytes부터)")
            elif existing == total_size:
                return  # already complete

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        # If server doesn't support Range, restart from scratch
        if resume and initial > 0 and resp.status_code != 206:
            mode = "wb"
            initial = 0

        pbar = vessel_progress(
            iterable=None,
            total=total_size,
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
DownloaderFactory.register("http", HTTPDownloader)
