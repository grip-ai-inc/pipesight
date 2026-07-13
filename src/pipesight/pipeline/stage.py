"""StageSpec: declares one stage of a Pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipesight.trace.schema import Device


@dataclass
class StageSpec:
    """One stage in a `Pipeline`.

    `device="gpu"` is a pure hint -- it never imports or touches any GPU
    framework in this module. It only affects (a) the default `workers`
    count (1, to avoid oversubscribing a single physical GPU unless you
    explicitly ask for more) and (b) how `pipesight.analysis` treats the
    resulting spans for idle-gap accounting.
    """

    name: str
    fn: Callable[[Any], Any]
    device: Device = "cpu"
    workers: int | None = None
    max_queue: int = 4
    item_id_fn: Callable[[Any], str | int] | None = None

    def resolved_workers(self) -> int:
        return self.workers if self.workers is not None else 1
