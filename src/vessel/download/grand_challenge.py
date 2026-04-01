"""Grand Challenge stub downloader.

Grand Challenge datasets require manual download through the web interface
because they need user authentication and Data Use Agreement (DUA)
acceptance.  This downloader provides instructions instead of actual
downloading.
"""

from __future__ import annotations

from pathlib import Path

from vessel.core.metadata import SourceConfig
from vessel.download.base import BaseDownloader, DownloaderFactory


class GrandChallengeDownloader(BaseDownloader):
    """Stub downloader that prints manual download instructions.

    Grand Challenge datasets cannot be automatically downloaded because
    they require:

    1. A registered account on grand-challenge.org
    2. Acceptance of a Data Use Agreement (DUA)
    3. Manual download through the web interface
    """

    def check_available(self) -> bool:
        """Always returns False — manual download required."""
        return False

    def download(self, resume: bool = True) -> list[Path]:
        """Print instructions for manual download and raise an error.

        Parameters
        ----------
        resume : bool
            Ignored; manual download required.

        Returns
        -------
        list[Path]
            Never returns — always raises.

        Raises
        ------
        RuntimeError
            Always raised with instructions.
        """
        url = self.source.url or "https://grand-challenge.org"

        instructions = (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║         Grand Challenge — 수동 다운로드 필요                ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║                                                            ║\n"
            "║  이 데이터셋은 자동 다운로드를 지원하지 않습니다.          ║\n"
            "║  아래 단계에 따라 수동으로 다운로드해 주세요:              ║\n"
            "║                                                            ║\n"
            "║  1. Grand Challenge 웹사이트에 계정을 만드세요             ║\n"
            "║  2. 데이터셋 페이지에서 DUA(Data Use Agreement)에          ║\n"
            "║     동의하세요                                             ║\n"
            "║  3. 데이터를 다운로드하세요                                ║\n"
            f"║  4. 다운로드한 파일을 아래 디렉토리에 넣으세요:           ║\n"
            "║                                                            ║\n"
            f"║  📂 {str(self.dest_dir):<52} ║\n"
            "║                                                            ║\n"
            f"║  🌐 {url:<52} ║\n"
            "║                                                            ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )

        self._log_progress(instructions)

        # Also list expected files if known
        if self.source.files:
            self._log_progress("예상 파일 목록:")
            for fi in self.source.files:
                self._log_progress(f"  - {fi.name}")

        raise RuntimeError(
            f"Grand Challenge 데이터셋은 수동 다운로드가 필요합니다. "
            f"웹사이트를 방문하세요: {url}"
        )


# Register with the factory
DownloaderFactory.register("grand_challenge", GrandChallengeDownloader)
