"""Pipeline: overlaps adjacent stages of a
`for item in items: stage_a(item); stage_b(item); ...`-shaped loop by
running each stage's function in its own small thread pool, connected by
bounded queues -- so stage N+1 can start on an item while stage N is still
working on the next one, instead of the whole loop blocking on every stage
in turn.

100% stdlib (threading/queue/dataclasses) -- no torch or CUDA import
anywhere in this module, so it works for any GPU framework or none at all.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from pipesight.pipeline.errors import PipelineClosedError, PipelineItemError
from pipesight.pipeline.stage import StageSpec
from pipesight.profiling.profiler import Profiler

logger = logging.getLogger(__name__)

_SENTINEL = object()
_Q_POLL_S = 0.1  # how often blocked queue ops re-check the stop event


class _ResultBox:
    __slots__ = ("seq_id", "value", "error")

    def __init__(self, seq_id: int, value: Any = None, error: BaseException | None = None) -> None:
        self.seq_id = seq_id
        self.value = value
        self.error = error


class _StageBoundary:
    """Coordinates sentinel-based shutdown between one stage's N worker
    threads and the M things reading its output (the next stage's M
    workers, or -- for the last stage -- the single `run()` consumer, so
    M=1 there). N and M need not match: whichever of this stage's workers
    is the *last* to see its sentinel is the one that forwards exactly M
    sentinels downstream."""

    def __init__(self, n_workers: int, n_downstream_readers: int) -> None:
        self._remaining = n_workers
        self._n_downstream_readers = n_downstream_readers
        self._lock = threading.Lock()

    def worker_done(self) -> int:
        with self._lock:
            self._remaining -= 1
            if self._remaining == 0:
                return self._n_downstream_readers
        return 0


def _put_stoppable(q: queue.Queue, value: Any, stop_event: threading.Event) -> bool:
    """Blocking put that periodically checks `stop_event` instead of
    blocking forever on a full queue. Returns False if stopped before the
    put succeeded."""
    while not stop_event.is_set():
        try:
            q.put(value, timeout=_Q_POLL_S)
            return True
        except queue.Full:
            continue
    return False


def _get_stoppable(q: queue.Queue, stop_event: threading.Event) -> Any:
    """Blocking get that periodically checks `stop_event`. Returns
    `_SENTINEL` if stopped before an item arrived (indistinguishable from a
    real sentinel to callers, which is the intent -- both mean "stop")."""
    while not stop_event.is_set():
        try:
            return q.get(timeout=_Q_POLL_S)
        except queue.Empty:
            continue
    return _SENTINEL


class Pipeline:
    """Use as a context manager:

        with Pipeline(stages, profiler=prof) as pipeline:
            for result in pipeline.run(items):
                handle(result)

    `close()` (or leaving the `with` block) stops every stage's worker
    threads and the feeder thread within one `_Q_POLL_S` tick, even if
    `run()`'s generator was abandoned mid-iteration.
    """

    def __init__(
        self,
        stages: list[StageSpec],
        *,
        profiler: Profiler | None = None,
        ordered: bool = True,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        self.stages = stages
        self.profiler = profiler
        self.ordered = ordered
        self.on_error = on_error

        self._worker_counts = [s.resolved_workers() for s in stages]
        self._queues: list[queue.Queue] = [queue.Queue(maxsize=s.max_queue) for s in stages]
        self._output_queue: queue.Queue = queue.Queue(maxsize=max(s.max_queue for s in stages))
        self._boundaries = [
            _StageBoundary(
                self._worker_counts[i],
                self._worker_counts[i + 1] if i + 1 < len(stages) else 1,
            )
            for i in range(len(stages))
        ]
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False

    # ---------- worker threads ----------

    def _stage_worker(self, stage_idx: int) -> None:
        stage = self.stages[stage_idx]
        in_q = self._queues[stage_idx]
        is_last = stage_idx == len(self.stages) - 1
        out_q = self._queues[stage_idx + 1] if not is_last else self._output_queue
        boundary = self._boundaries[stage_idx]

        while True:
            entry = _get_stoppable(in_q, self._stop_event)
            if entry is _SENTINEL:
                if self._stop_event.is_set():
                    return  # shutting down; don't bother forwarding sentinels
                n_sentinels = boundary.worker_done()
                for _ in range(n_sentinels):
                    _put_stoppable(out_q, _SENTINEL, self._stop_event)
                return

            seq_id, payload = entry
            item_id = stage.item_id_fn(payload) if stage.item_id_fn else seq_id
            try:
                if self.profiler is not None:
                    with self.profiler.stage(stage.name, device=stage.device, item_id=item_id):
                        result = stage.fn(payload)
                else:
                    result = stage.fn(payload)
            except Exception as exc:  # noqa: BLE001 -- broad on purpose; surfaced via PipelineItemError
                error = PipelineItemError(seq_id, stage.name, exc)
                box = _ResultBox(seq_id, error=error)
                _put_stoppable(self._output_queue, box, self._stop_event)
                continue

            box_or_tuple = _ResultBox(seq_id, value=result) if is_last else (seq_id, result)
            _put_stoppable(out_q, box_or_tuple, self._stop_event)

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        for stage_idx, count in enumerate(self._worker_counts):
            for _ in range(count):
                t = threading.Thread(target=self._stage_worker, args=(stage_idx,), daemon=True)
                t.start()
                self._threads.append(t)

    def _feed(self, items: Iterable[Any], total_holder: list[int]) -> None:
        seq_id = -1
        for seq_id, item in enumerate(items):
            if self._stop_event.is_set():
                break
            if not _put_stoppable(self._queues[0], (seq_id, item), self._stop_event):
                break
        for _ in range(self._worker_counts[0]):
            _put_stoppable(self._queues[0], _SENTINEL, self._stop_event)
        total_holder.append(seq_id + 1)

    # ---------- public API ----------

    def run(self, items: Iterable[Any]) -> Iterator[Any]:
        if self._closed:
            raise PipelineClosedError("Pipeline is closed")
        self._start()

        total_holder: list[int] = []
        feeder = threading.Thread(target=self._feed, args=(items, total_holder), daemon=True)
        feeder.start()

        pending: dict[int, _ResultBox] = {}
        next_expected = 0
        received = 0
        total: int | None = None

        while total is None or received < total:
            entry = _get_stoppable(self._output_queue, self._stop_event)
            if entry is _SENTINEL:
                if self._stop_event.is_set():
                    return
                feeder.join()
                total = total_holder[0]
                continue

            box: _ResultBox = entry
            received += 1
            if not self.ordered:
                if box.error is not None:
                    if self.on_error == "raise":
                        raise box.error
                    logger.warning("skipping item #%d: %s", box.seq_id, box.error)
                    continue
                yield box.value
                continue

            pending[box.seq_id] = box
            while next_expected in pending:
                ready = pending.pop(next_expected)
                next_expected += 1
                if ready.error is not None:
                    if self.on_error == "raise":
                        raise ready.error
                    logger.warning("skipping item #%d: %s", ready.seq_id, ready.error)
                    continue
                yield ready.value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
