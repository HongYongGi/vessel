"""Vessel download module — downloaders for various dataset sources.

Importing this package registers all built-in downloaders with the
:class:`~vessel.download.base.DownloaderFactory`.
"""

from vessel.download.base import BaseDownloader, DownloaderFactory
from vessel.download.extract import extract_archive

# Import each downloader module to trigger its DownloaderFactory.register() call.
import vessel.download.zenodo  # noqa: F401
import vessel.download.kaggle  # noqa: F401
import vessel.download.gdrive  # noqa: F401
import vessel.download.http  # noqa: F401
import vessel.download.grand_challenge  # noqa: F401

__all__ = [
    "BaseDownloader",
    "DownloaderFactory",
    "extract_archive",
]
