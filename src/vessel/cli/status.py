"""CLI command for showing download / preprocess status."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vessel.core.paths import get_status_dir

status_app = typer.Typer(help="Show download / preprocess status")
console = Console()


def _load_status(filename: str) -> dict:
    """Load a status JSON file from the .vessel metadata directory."""
    try:
        status_path = get_status_dir() / filename
    except EnvironmentError:
        return {}  # VESSEL_DATA_ROOT not set
    if not status_path.exists():
        return {}
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_size(size_bytes: int | float | None) -> str:
    """Human-readable file size."""
    if size_bytes is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} PB"


def status() -> None:
    """Display download and preprocess status for all datasets."""
    download_status = _load_status("download_status.json")
    preprocess_status = _load_status("preprocess_status.json")

    # Merge all known dataset IDs
    all_ids = sorted(set(download_status.keys()) | set(preprocess_status.keys()))

    if not all_ids:
        console.print("[yellow]No status information found.[/yellow]")
        try:
            console.print(f"[dim]Looked in: {get_status_dir()}[/dim]")
        except EnvironmentError:
            console.print("[dim]VESSEL_DATA_ROOT is not set.[/dim]")
        raise typer.Exit()

    table = Table(title="Dataset Status", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Downloaded", justify="center")
    table.add_column("Preprocessed", justify="center")
    table.add_column("Size", justify="right", style="dim")

    for ds_id in all_ids:
        dl = download_status.get(ds_id, {})
        pp = preprocess_status.get(ds_id, {})

        name = dl.get("name", pp.get("name", ds_id))

        dl_done = dl.get("completed", False)
        pp_done = pp.get("completed", False)

        dl_icon = "[green]✓[/green]" if dl_done else "[red]✗[/red]"
        pp_icon = "[green]✓[/green]" if pp_done else "[red]✗[/red]"

        size = _format_size(dl.get("size_bytes"))

        table.add_row(ds_id, name, dl_icon, pp_icon, size)

    console.print(table)


# Register under the sub-app as well
status_app.command(name="show")(status)
