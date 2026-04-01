"""CLI commands for exporting datasets to nnUNet v2 format.

Usage examples::

    vessel export nnunet aortaseg24 --task-id 500
    vessel export nnunet aortaseg24 --task-id 500 --task-name AortaOnly
    vessel export nnunet aortaseg24 --task-id 500 --labels ascending_aorta,aortic_arch
    vessel export nnunet --merge aortaseg24,topcow --task-id 501 --task-name AllVessel
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

export_app = typer.Typer(
    name="export",
    help="Export datasets to training-ready formats (nnUNet, etc.)",
    no_args_is_help=True,
)

console = Console()


@export_app.command("nnunet")
def export_nnunet(
    dataset_id: Optional[str] = typer.Argument(  # noqa: UP007
        None,
        help="Dataset ID to export (e.g. aortaseg24). Omit when using --merge.",
    ),
    task_id: int = typer.Option(
        ...,
        "--task-id",
        help="nnUNet task/dataset ID (e.g. 500).",
    ),
    task_name: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--task-name",
        help="Human-readable task name. Defaults to dataset name.",
    ),
    labels: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--labels",
        help="Comma-separated list of label names to include (subset).",
    ),
    merge: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--merge",
        help="Comma-separated dataset IDs to merge into one nnUNet dataset.",
    ),
    export_base: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--export-base",
        help="Override export base directory (default: $VESSEL_DATA_ROOT/exports/nnUNet_raw).",
    ),
) -> None:
    """Export dataset(s) to nnUNet v2 raw format."""
    from vessel.export.nnunet import NnUNetExporter

    label_subset = [s.strip() for s in labels.split(",")] if labels else None
    base_path = Path(export_base) if export_base else None

    exporter = NnUNetExporter(export_base=base_path)

    if merge:
        # Merge mode
        dataset_ids = [s.strip() for s in merge.split(",")]
        if task_name is None:
            console.print(
                "[red]Error:[/] --task-name is required when using --merge."
            )
            raise typer.Exit(code=1)
        console.print(
            f"[bold]Merging {len(dataset_ids)} datasets into "
            f"nnUNet Dataset{task_id:03d}_{task_name}...[/]"
        )
        result_dir = exporter.export_merged(
            dataset_ids=dataset_ids,
            task_id=task_id,
            task_name=task_name,
            label_subset=label_subset,
        )
    elif dataset_id:
        # Single dataset mode
        console.print(
            f"[bold]Exporting '{dataset_id}' as nnUNet Dataset{task_id:03d}...[/]"
        )
        result_dir = exporter.export_single(
            dataset_id=dataset_id,
            task_id=task_id,
            task_name=task_name,
            label_subset=label_subset,
        )
    else:
        console.print(
            "[red]Error:[/] Provide a dataset_id or use --merge."
        )
        raise typer.Exit(code=1)

    console.print(f"[green]Done![/] Output: {result_dir}")
