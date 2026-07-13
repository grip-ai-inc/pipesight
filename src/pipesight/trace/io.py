"""Chrome Trace Event Format JSON codec for `Trace`.

Spans are written as standard "X" (complete event) / "i" (instant event)
entries under `traceEvents`, so any trace pipesight writes opens directly in
chrome://tracing or ui.perfetto.dev with no conversion. A sibling top-level
`"pipesight"` key carries metadata and coarse utilization samples that those
viewers don't understand but silently ignore -- one file is simultaneously a
valid Chrome trace and a self-contained pipesight artifact.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from pipesight.trace.schema import Device, Sample, Span, Trace, TraceMeta

_SCHEMA_VERSION = 1


def _span_to_event(span: Span) -> dict:
    args = dict(span.args)
    if span.item_id is not None:
        args["item_id"] = span.item_id
    if span.worker_id is not None:
        args["worker_id"] = span.worker_id

    event = {
        "name": span.name,
        "cat": span.device,
        "pid": span.proc_id,
        "tid": span.thread_id,
        "ts": span.start_ns / 1000.0,
    }
    if span.end_ns == span.start_ns:
        event["ph"] = "i"
        event["s"] = "t"  # thread-scoped instant event
    else:
        event["ph"] = "X"
        event["dur"] = span.duration_ns / 1000.0
    if args:
        event["args"] = args
    return event


def _event_to_span(event: dict) -> Span | None:
    if event.get("ph") not in ("X", "i"):
        return None
    args = dict(event.get("args", {}))
    item_id = args.pop("item_id", None)
    worker_id = args.pop("worker_id", None)
    start_ns = round(event["ts"] * 1000)
    dur_us = event.get("dur", 0.0)
    end_ns = start_ns + round(dur_us * 1000)
    device: Device = event.get("cat", "other")  # type: ignore[assignment]
    return Span(
        name=event["name"],
        device=device,
        start_ns=start_ns,
        end_ns=end_ns,
        proc_id=event.get("pid", 0),
        thread_id=event.get("tid", 0),
        item_id=item_id,
        worker_id=worker_id,
        args=args,
    )


def _sample_to_dict(sample: Sample) -> dict:
    return {
        "ts_ns": sample.ts_ns,
        "cpu_percent": sample.cpu_percent,
        "gpu_util_pct": sample.gpu_util_pct,
        "gpu_mem_used_mb": sample.gpu_mem_used_mb,
        "proc_id": sample.proc_id,
    }


def _dict_to_sample(d: dict) -> Sample:
    return Sample(
        ts_ns=d["ts_ns"],
        cpu_percent=d.get("cpu_percent", []),
        gpu_util_pct=d.get("gpu_util_pct"),
        gpu_mem_used_mb=d.get("gpu_mem_used_mb"),
        proc_id=d.get("proc_id"),
    )


def to_dict(trace: Trace) -> dict:
    return {
        "traceEvents": [_span_to_event(s) for s in trace.spans],
        "pipesight": {
            "schema_version": _SCHEMA_VERSION,
            "meta": dataclasses.asdict(trace.meta),
            "samples": [_sample_to_dict(s) for s in trace.samples],
        },
    }


def from_dict(data: dict) -> Trace:
    pipesight_block = data.get("pipesight", {})
    meta_dict = pipesight_block.get("meta", {})
    meta = TraceMeta(**{k: v for k, v in meta_dict.items() if k in TraceMeta.__dataclass_fields__})
    spans = [s for s in (_event_to_span(e) for e in data.get("traceEvents", [])) if s is not None]
    samples = [_dict_to_sample(d) for d in pipesight_block.get("samples", [])]
    return Trace(meta=meta, spans=spans, samples=samples)


def save(trace: Trace, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(trace), indent=None))


def load(path: str | Path) -> Trace:
    data = json.loads(Path(path).read_text())
    return from_dict(data)


def _epoch_ns(meta: TraceMeta, t_ns: int) -> int:
    return round(meta.wall_start_epoch_s * 1e9) + (t_ns - meta.perf_counter_offset_ns)


def merge(traces: list[Trace]) -> Trace:
    """Combine per-process traces into one, realigning each process's
    monotonic clock onto a shared wall-clock timeline via
    `wall_start_epoch_s`/`perf_counter_offset_ns` (perf_counter_ns() values
    are only comparable within a single process)."""
    if not traces:
        raise ValueError("merge() requires at least one trace")
    if len(traces) == 1:
        return traces[0]

    epoch_starts = [_epoch_ns(t.meta, s.start_ns) for t in traces for s in t.spans]
    epoch_starts += [_epoch_ns(t.meta, sm.ts_ns) for t in traces for sm in t.samples]
    default_base_ns = round(traces[0].meta.wall_start_epoch_s * 1e9)
    base_epoch_ns = min(epoch_starts) if epoch_starts else default_base_ns

    def remap(t_ns: int, meta: TraceMeta) -> int:
        return _epoch_ns(meta, t_ns) - base_epoch_ns

    merged_spans = [
        dataclasses.replace(
            s, start_ns=remap(s.start_ns, t.meta), end_ns=remap(s.end_ns, t.meta)
        )
        for t in traces
        for s in t.spans
    ]
    merged_samples = [
        dataclasses.replace(sm, ts_ns=remap(sm.ts_ns, t.meta)) for t in traces for sm in t.samples
    ]
    ref = traces[0].meta
    merged_meta = TraceMeta(
        schema_version=_SCHEMA_VERSION,
        source=ref.source,
        hostname=ref.hostname,
        gpu_name=ref.gpu_name,
        cpu_count_logical=ref.cpu_count_logical,
        cpu_count_physical=ref.cpu_count_physical,
        command=ref.command,
        wall_start_epoch_s=base_epoch_ns / 1e9,
        perf_counter_offset_ns=0,
    )
    return Trace(meta=merged_meta, spans=merged_spans, samples=merged_samples)


def load_many(paths: list[str | Path]) -> list[Trace]:
    return [load(p) for p in paths]
