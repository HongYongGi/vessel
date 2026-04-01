"""Vessel CLI - dataset download & preprocessing tool."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="vessel",
    help="Vessel segmentation dataset download & preprocessing tool",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Vessel segmentation dataset management tool."""


# ---------------------------------------------------------------------------
# Register sub-command groups
# ---------------------------------------------------------------------------

from vessel.cli.registry import list_datasets, info_dataset, registry_app  # noqa: E402
from vessel.cli.status import status, status_app  # noqa: E402

app.add_typer(registry_app, name="registry", help="List & inspect registered datasets")
app.add_typer(status_app, name="status", help="Show download / preprocess status")

# Top-level convenience aliases
app.command(name="list")(list_datasets)
app.command(name="info")(info_dataset)
app.command(name="show-status")(status)

# Optional modules — only register if their dependencies exist
try:
    from vessel.cli.download import download_app  # noqa: E402
    app.add_typer(download_app, name="download", help="Download datasets")
except ImportError:
    pass

try:
    from vessel.cli.preprocess import preprocess_app  # noqa: E402
    app.add_typer(preprocess_app, name="preprocess", help="Preprocess datasets")
except ImportError:
    pass

try:
    from vessel.cli.export import export_app  # noqa: E402
    app.add_typer(export_app, name="export", help="Export to nnUNet format")
except ImportError:
    pass
