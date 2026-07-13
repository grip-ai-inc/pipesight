"""Errors raised by `Pipeline`."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for pipeline-related errors."""


class PipelineItemError(PipelineError):
    """Wraps an exception a stage function raised while processing one item.

    Surfaces in the caller's own iteration of `Pipeline.run()` (at the point
    the caller would have received that item), never silently inside a
    background worker thread.
    """

    def __init__(self, seq_id: int, stage_name: str, original: BaseException) -> None:
        self.seq_id = seq_id
        self.stage_name = stage_name
        self.original = original
        super().__init__(f"stage {stage_name!r} failed on item #{seq_id}: {original!r}")


class PipelineClosedError(PipelineError):
    """Raised when trying to `run()` a `Pipeline` that has already been closed."""
