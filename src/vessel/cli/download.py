"""CLI commands for downloading datasets.

Usage::

    vessel download <dataset_id> [--resume] [--dry-run]
    vessel download --all [--tier N]
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

download_app = typer.Typer(help="Download datasets")
console = Console()


def _get_registry():
    """Lazy import to avoid heavy imports at CLI startup."""
    from vessel.core.registry import DatasetRegistry

    return DatasetRegistry()


def _get_status_path() -> Path:
    """Return path to the download status JSON file."""
    from vessel.core.paths import get_status_dir

    status_dir = get_status_dir()
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir / "download_status.json"


def _load_status() -> dict:
    """Load existing download status."""
    path = _get_status_path()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_status(status: dict) -> None:
    """Save download status to disk."""
    path = _get_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def _download_single(
    dataset_id: str,
    *,
    resume: bool = True,
    dry_run: bool = False,
) -> bool:
    """Download a single dataset. Returns True on success."""
    from vessel.core.paths import get_raw_dir, ensure_dirs
    from vessel.download.base import DownloaderFactory
    from vessel.download.extract import extract_archive

    reg = _get_registry()

    try:
        config = reg.get(dataset_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        return False

    source = config.source
    dest_dir = get_raw_dir(dataset_id)

    console.print(
        Panel(
            f"[bold]{config.dataset.name}[/bold]\n"
            f"소스 타입: {source.type}\n"
            f"예상 크기: {config.dataset.estimated_size_gb:.1f} GB\n"
            f"대상 경로: {dest_dir}",
            title=f"다운로드: {dataset_id}",
            border_style="cyan",
        )
    )

    if dry_run:
        console.print("[yellow]--dry-run: 실제 다운로드를 건너뜁니다.[/yellow]")
        return True

    ensure_dirs(dataset_id)

    # Create the appropriate downloader
    try:
        downloader = DownloaderFactory.create(source, dest_dir)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return False

    # Check availability
    if not downloader.check_available():
        console.print(
            f"[yellow]경고: {source.type} 소스에 접근할 수 없습니다.[/yellow]"
        )
        if source.type == "grand_challenge":
            try:
                downloader.download(resume=resume)
            except RuntimeError as e:
                console.print(f"[yellow]{e}[/yellow]")
            return False
        console.print("[red]다운로드를 건너뜁니다.[/red]")
        return False

    # Download
    start_time = time.time()
    try:
        downloaded_files = downloader.download(resume=resume)
    except Exception as e:
        console.print(f"[red]다운로드 실패: {e}[/red]")
        return False

    elapsed = time.time() - start_time
    console.print(
        f"[green]다운로드 완료![/green] "
        f"({len(downloaded_files)} 파일, {elapsed:.1f}초)"
    )

    # Extract archives if needed
    if source.extract != "none":
        console.print(f"[cyan]압축 해제 중 (형식: {source.extract})...[/cyan]")
        for fpath in downloaded_files:
            try:
                extract_archive(fpath, dest_dir, format=source.extract)
                console.print(f"[green]압축 해제 완료: {fpath.name}[/green]")
            except Exception as e:
                console.print(f"[red]압축 해제 실패: {fpath.name} — {e}[/red]")
                return False

    # Update status
    status = _load_status()
    total_size = sum(
        f.stat().st_size for f in downloaded_files if f.exists()
    )
    status[dataset_id] = {
        "name": config.dataset.name,
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "size_bytes": total_size,
        "num_files": len(downloaded_files),
        "source_type": source.type,
    }
    _save_status(status)

    return True


@download_app.command(name="one")
def download_one(
    dataset_id: str = typer.Argument(..., help="Dataset identifier"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume partial downloads."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded."),
) -> None:
    """Download a single dataset."""
    success = _download_single(dataset_id, resume=resume, dry_run=dry_run)
    if not success:
        raise typer.Exit(code=1)


@download_app.command(name="all")
def download_all_cmd(
    tier: Optional[int] = typer.Option(None, "--tier", help="Only download datasets of this tier."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume partial downloads."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded."),
) -> None:
    """Download all (or tier-filtered) datasets."""
    _download_all(tier=tier, resume=resume, dry_run=dry_run)


def _download_all(
    *,
    tier: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> None:
    """Download all (or tier-filtered) datasets."""
    reg = _get_registry()
    configs = reg.list_datasets(tier=tier)

    if not configs:
        console.print("[yellow]조건에 맞는 데이터셋이 없습니다.[/yellow]")
        raise typer.Exit()

    total = len(configs)
    console.print(f"[bold]총 {total}개 데이터셋 다운로드 시작[/bold]\n")

    succeeded = 0
    failed = 0

    for idx, cfg in enumerate(configs, 1):
        ds_id = cfg.dataset.id
        console.print(
            f"\n[bold cyan]━━━ [{idx}/{total}] {ds_id} ━━━[/bold cyan]"
        )
        ok = _download_single(ds_id, resume=resume, dry_run=dry_run)
        if ok:
            succeeded += 1
        else:
            failed += 1

    console.print(f"\n[bold]결과: {succeeded} 성공, {failed} 실패 (총 {total})[/bold]")
    if failed > 0:
        raise typer.Exit(code=1)
