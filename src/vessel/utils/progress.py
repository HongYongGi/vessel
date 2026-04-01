"""tqdm-based progress bar wrapper with vessel's preferred format.

Display format:
    [현재/전체] ████████░░ 80% | 경과: 00:30 | 남은시간: ~00:08 | 현재작업
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")

# Custom bar format matching the user's preferred style.
# tqdm format variables: n, total, bar, percentage, elapsed, remaining, desc
_BAR_FMT = (
    "[{n_fmt}/{total_fmt}] {bar} {percentage:3.0f}%"
    " | 경과: {elapsed}"
    " | 남은시간: ~{remaining}"
    " | {desc}"
)


def vessel_progress(
    iterable: Iterable[T] | None = None,
    total: int | None = None,
    desc: str = "Processing",
    *,
    leave: bool = True,
    ncols: int | None = 100,
    **kwargs,
) -> tqdm[T]:
    """Return a tqdm iterator with vessel's custom progress bar format.

    Parameters
    ----------
    iterable : Iterable[T] | None
        The iterable to wrap.  Can be ``None`` for manual ``.update()`` usage.
    total : int | None
        Total number of items.  Inferred from *iterable* if possible.
    desc : str
        Short description shown at the right end of the bar.
    leave : bool
        Whether to leave the bar on screen after completion.
    ncols : int | None
        Fixed width of the bar (``None`` for auto-detect).

    Returns
    -------
    tqdm[T]
        A tqdm progress bar object that can be iterated or updated manually.

    Example
    -------
    >>> for item in vessel_progress(range(100), desc="Loading scans"):
    ...     process(item)
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        bar_format=_BAR_FMT,
        leave=leave,
        ncols=ncols,
        ascii="░█",
        **kwargs,
    )


@contextlib.contextmanager
def vessel_progress_ctx(
    total: int,
    desc: str = "Processing",
    **kwargs,
) -> Iterator[tqdm]:
    """Context manager version for manual progress updates.

    Example
    -------
    >>> with vessel_progress_ctx(total=50, desc="Downloading") as pbar:
    ...     for chunk in stream:
    ...         write(chunk)
    ...         pbar.update(1)
    ...         pbar.set_description(f"chunk {chunk.name}")
    """
    pbar = vessel_progress(iterable=None, total=total, desc=desc, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()
