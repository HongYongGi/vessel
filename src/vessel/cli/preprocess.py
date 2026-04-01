"""CLI commands for dataset preprocessing."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

preprocess_app = typer.Typer(help="Preprocess datasets")
console = Console()


def _get_registry():
    """Lazy import to avoid heavy imports at CLI startup."""
    from vessel.core.registry import DatasetRegistry  # noqa: WPS433

    return DatasetRegistry()


def _run_pipeline(dataset_id: str, workers: int) -> dict:
    """Instantiate and run the preprocessing pipeline for one dataset."""
    from vessel.preprocess.pipeline import PreprocessPipeline  # noqa: WPS433

    reg = _get_registry()
    try:
        cfg = reg.get(dataset_id)
    except KeyError:
        console.print(f"[red]Dataset '{dataset_id}' not found in registry.[/red]")
        raise typer.Exit(code=1)

    pipeline = PreprocessPipeline(cfg)
    return pipeline.run(workers=workers)


def _print_report(dataset_id: str, report: dict) -> None:
    """Pretty-print a pipeline execution report."""
    # Summary panel
    status_color = "green" if report["failed"] == 0 else "yellow"
    summary_lines = [
        f"[bold]Dataset:[/bold]  {dataset_id}",
        f"[bold]Total:[/bold]    {report['total']}",
        f"[bold]Success:[/bold]  [green]{report['success']}[/green]",
        f"[bold]Failed:[/bold]   [red]{report['failed']}[/red]",
        f"[bold]Elapsed:[/bold]  {report['elapsed_sec']}s",
    ]
    if report.get("splits_path"):
        summary_lines.append(f"[bold]Splits:[/bold]  {report['splits_path']}")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title=f"Preprocessing Report: {dataset_id}",
            border_style=status_color,
            expand=False,
        )
    )

    # Errors
    errors = report.get("errors", {})
    if errors:
        console.print("\n[bold red]Processing Errors:[/bold red]")
        for case_id, err in sorted(errors.items()):
            console.print(f"  [red]{case_id}[/red]: {err}")

    # Validation issues
    validation = report.get("validation", {})
    issues = {k: v for k, v in validation.items() if v}
    if issues:
        console.print("\n[bold yellow]Validation Issues:[/bold yellow]")
        for case_id, issue_list in sorted(issues.items()):
            for issue in issue_list:
                console.print(f"  [yellow]{case_id}[/yellow]: {issue}")
    elif report["success"] > 0:
        console.print("\n[green]All cases passed validation.[/green]")


@preprocess_app.command(name="run")
def preprocess_run(
    dataset_id: str = typer.Argument(..., help="Dataset identifier (e.g. 'msd_hepatic_vessel')"),
    workers: int = typer.Option(1, "--workers", help="Number of parallel workers"),
    steps: Optional[str] = typer.Option(
        None,
        "--steps",
        help="Comma-separated list of steps to run (reserved for future use)",
    ),
) -> None:
    """Preprocess a single dataset.

    Example::

        vessel preprocess run msd_hepatic_vessel --workers 4
    """
    if steps:
        console.print(f"[dim]Steps filter: {steps} (not yet implemented, running all)[/dim]")

    console.print(f"\n[bold cyan]Starting preprocessing: {dataset_id}[/bold cyan]\n")
    report = _run_pipeline(dataset_id, workers=workers)
    _print_report(dataset_id, report)


@preprocess_app.command(name="all")
def preprocess_all(
    tier: Optional[int] = typer.Option(None, "--tier", help="Only preprocess datasets of this tier"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers per dataset"),
) -> None:
    """Preprocess all (or tier-filtered) datasets.

    Example::

        vessel preprocess all --tier 1 --workers 4
    """
    reg = _get_registry()
    datasets = reg.list_datasets(tier=tier)

    if not datasets:
        console.print("[yellow]No datasets matched the given filters.[/yellow]")
        raise typer.Exit()

    ids = [cfg.dataset.id for cfg in datasets]
    console.print(
        f"\n[bold cyan]Preprocessing {len(ids)} dataset(s):[/bold cyan] "
        + ", ".join(ids)
        + "\n"
    )

    all_reports: dict[str, dict] = {}
    for ds_id in ids:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]{ds_id}[/bold]")
        console.print(f"{'='*60}")

        report = _run_pipeline(ds_id, workers=workers)
        all_reports[ds_id] = report
        _print_report(ds_id, report)

    # Final summary table
    console.print(f"\n{'='*60}")
    console.print("[bold]Overall Summary[/bold]")
    console.print(f"{'='*60}\n")

    table = Table(show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Time (s)", justify="right")

    for ds_id, rpt in all_reports.items():
        table.add_row(
            ds_id,
            str(rpt["total"]),
            str(rpt["success"]),
            str(rpt["failed"]),
            str(rpt["elapsed_sec"]),
        )

    console.print(table)


@preprocess_app.command(name="validate")
def preprocess_validate(
    dataset_id: str = typer.Argument(..., help="Dataset identifier"),
) -> None:
    """Validate a previously processed dataset.

    Example::

        vessel preprocess validate msd_hepatic_vessel
    """
    from vessel.core.paths import get_processed_dir  # noqa: WPS433
    from vessel.preprocess.validate import validate_dataset  # noqa: WPS433

    processed_dir = get_processed_dir(dataset_id)
    if not processed_dir.is_dir():
        console.print(
            f"[red]Processed directory not found: {processed_dir}[/red]\n"
            "Run preprocessing first."
        )
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Validating: {dataset_id}[/bold cyan]\n")
    results = validate_dataset(processed_dir)

    issues = {k: v for k, v in results.items() if v}
    if not issues:
        console.print(f"[green]All cases in '{dataset_id}' passed validation.[/green]")
    else:
        console.print(f"[yellow]{len(issues)} case(s) with issues:[/yellow]\n")
        for case_id, issue_list in sorted(issues.items()):
            for issue in issue_list:
                console.print(f"  [yellow]{case_id}[/yellow]: {issue}")
