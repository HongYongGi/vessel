"""Abstract base downloader and factory for creating source-specific downloaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from vessel.core.metadata import SourceConfig


class BaseDownloader(ABC):
    """Base class for all dataset downloaders.

    Parameters
    ----------
    source : SourceConfig
        The source configuration describing where to download from.
    dest_dir : Path
        Local directory to save downloaded files into.
    """

    def __init__(self, source: SourceConfig, dest_dir: Path) -> None:
        self.source = source
        self.dest_dir = dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self, resume: bool = True) -> list[Path]:
        """Download all files from the source.

        Parameters
        ----------
        resume : bool
            If True, attempt to resume partially downloaded files.

        Returns
        -------
        list[Path]
            List of paths to downloaded files.
        """
        ...

    @abstractmethod
    def check_available(self) -> bool:
        """Check whether this download source is accessible.

        Returns
        -------
        bool
            True if the source can be reached / credentials are available.
        """
        ...

    def _log_progress(self, msg: str) -> None:
        """Print a styled progress message to the console."""
        from rich.console import Console

        Console().print(f"[cyan]{msg}[/cyan]")


class DownloaderFactory:
    """Factory that creates the appropriate downloader based on source type.

    Downloaders register themselves via :meth:`register`.  The factory
    looks up the registered class by ``source.type`` and instantiates it.
    """

    _registry: dict[str, type[BaseDownloader]] = {}

    @classmethod
    def register(cls, source_type: str, downloader_cls: type[BaseDownloader]) -> None:
        """Register a downloader class for a given source type.

        Parameters
        ----------
        source_type : str
            The source type identifier (e.g. ``"zenodo"``, ``"kaggle"``).
        downloader_cls : type[BaseDownloader]
            The downloader class to instantiate for this source type.
        """
        cls._registry[source_type] = downloader_cls

    @classmethod
    def create(cls, source: SourceConfig, dest_dir: Path) -> BaseDownloader:
        """Create a downloader instance for the given source config.

        Parameters
        ----------
        source : SourceConfig
            Source configuration with a ``type`` field.
        dest_dir : Path
            Destination directory for downloads.

        Returns
        -------
        BaseDownloader
            An instance of the registered downloader class.

        Raises
        ------
        ValueError
            If the source type has not been registered.
        """
        if source.type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise ValueError(
                f"Unknown source type: '{source.type}'. "
                f"Registered types: {available}"
            )
        return cls._registry[source.type](source, dest_dir)
