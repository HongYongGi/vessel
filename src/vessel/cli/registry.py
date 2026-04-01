"""CLI commands for listing and inspecting registered datasets."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

registry_app = typer.Typer(help="List & inspect registered datasets")
console = Console()


def _get_registry():
    from vessel.core.registry import DatasetRegistry
    return DatasetRegistry()


def list_datasets(
    tier: Optional[int] = typer.Option(None, "--tier", help="Filter by tier (1-3)"),
    region: Optional[str] = typer.Option(None, "--region", help="Filter by body region"),
) -> None:
    """Show a table of all registered datasets."""
    reg = _get_registry()
    datasets = reg.list_datasets(tier=tier, body_region=region)

    if not datasets:
        console.print("[yellow]No datasets matched the given filters.[/yellow]")
        raise typer.Exit()

    table = Table(title="Registered Datasets", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Tier", justify="center")
    table.add_column("Region", style="green")
    table.add_column("Modality", style="magenta")
    table.add_column("Cases", justify="right")
    table.add_column("Size (GB)", justify="right")

    for cfg in datasets:
        ds = cfg.dataset
        table.add_row(
            ds.id,
            ds.name,
            str(ds.tier),
            ds.body_region,
            ds.modality,
            str(ds.num_cases),
            f"{ds.estimated_size_gb:.1f}",
        )

    console.print(table)


def info_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset identifier"),
) -> None:
    """Show detailed information for a single dataset."""
    reg = _get_registry()

    try:
        cfg = reg.get(dataset_id)
    except KeyError:
        console.print(f"[red]Dataset '{dataset_id}' not found.[/red]")
        raise typer.Exit(code=1)

    ds = cfg.dataset
    src = cfg.source
    lbl = cfg.labels
    pp = cfg.preprocess

    lines = [
        f"[bold]{'ID':<20}[/bold]: {ds.id}",
        f"[bold]{'Name':<20}[/bold]: {ds.name}",
        f"[bold]{'Description':<20}[/bold]: {ds.description}",
        f"[bold]{'Tier':<20}[/bold]: {ds.tier}",
        f"[bold]{'Body Region':<20}[/bold]: {ds.body_region}",
        f"[bold]{'Modality':<20}[/bold]: {ds.modality}",
        f"[bold]{'Cases':<20}[/bold]: {ds.num_cases}",
        f"[bold]{'Est. Size (GB)':<20}[/bold]: {ds.estimated_size_gb}",
        f"[bold]{'License':<20}[/bold]: {ds.license}",
        f"[bold]{'Paper':<20}[/bold]: {ds.paper}",
        "",
        f"[bold]{'Source Type':<20}[/bold]: {src.type}",
        f"[bold]{'Extract':<20}[/bold]: {src.extract}",
        "",
        f"[bold]{'Label Type':<20}[/bold]: {lbl.type}",
        f"[bold]{'Num Classes':<20}[/bold]: {lbl.num_classes}",
        f"[bold]{'Label Mapping':<20}[/bold]:",
    ]
    for k, v in lbl.mapping.items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        f"[bold]{'Target Spacing':<20}[/bold]: {pp.target_spacing}",
        f"[bold]{'HU Window':<20}[/bold]: center={pp.intensity_window.center}, width={pp.intensity_window.width}",
        f"[bold]{'Normalize':<20}[/bold]: {pp.normalize}",
    ])

    panel = Panel(
        "\n".join(lines),
        title=f"Dataset: {ds.name}",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


registry_app.command(name="list")(list_datasets)
registry_app.command(name="info")(info_dataset)
