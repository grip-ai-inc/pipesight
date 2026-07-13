# Trace file schema

A pipesight trace file is a JSON document that is **simultaneously**:

1. A valid [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
   file — open it directly in `chrome://tracing` or
   [ui.perfetto.dev](https://ui.perfetto.dev) with zero conversion.
2. A self-contained pipesight artifact — `pipesight.trace.io.load()` reads
   it back into a `Trace` object with no information loss.

This works because pipesight's own metadata and utilization samples live
under one extra top-level key, `"pipesight"`, which Chrome's and Perfetto's
trace viewers simply ignore (both tolerate unrecognized top-level keys).

## Top-level structure

```json
{
  "traceEvents": [ ... ],
  "pipesight": {
    "schema_version": 1,
    "meta": { ... },
    "samples": [ ... ]
  }
}
```

## `traceEvents` — one entry per `Span`

Each `Span` (see `pipesight.trace.schema.Span`) is written as one Chrome
trace event. A real example, from `examples/synthetic_pipeline`:

```json
{
  "name": "gpu_compute",
  "cat": "gpu",
  "pid": 2108639,
  "tid": 137734944203072,
  "ts": 27497791238.022,
  "ph": "X",
  "dur": 119033.275,
  "args": {
    "gpu_busy_ns": 118884354,
    "timing_method": "cuda_event",
    "item_id": 0
  }
}
```

| Field | Chrome meaning | pipesight meaning |
|---|---|---|
| `name` | event name | the stage name passed to `profiler.stage(name, ...)` |
| `cat` | category | `Span.device`: `"cpu"`, `"gpu"`, or `"other"` |
| `pid` / `tid` | process/thread ID | `Span.proc_id` / `Span.thread_id` (`os.getpid()` / `threading.get_ident()`) |
| `ts` | start timestamp, **microseconds** | `Span.start_ns / 1000` |
| `ph` | event phase | `"X"` (complete event, has `dur`) for a timed stage; `"i"` (instant event, no `dur`) for a zero-duration `profiler.mark()` |
| `dur` | duration, **microseconds** | `Span.duration_ns / 1000` (omitted for instant events) |
| `args` | free-form key/value | `Span.item_id` (if set), `Span.worker_id` (if set), and everything in `Span.args` — e.g. `gpu_busy_ns` / `timing_method` when a `device="gpu"` stage was timed with `torch.cuda.Event`, or `error` when a stage raised |

**Units**: `Span` stores nanoseconds (`start_ns`, `end_ns`,
`duration_ns` — matching `time.perf_counter_ns()`, the clock everything is
measured with); the on-disk format uses **microseconds** for `ts`/`dur`
because that's the Chrome Trace Format's convention. `trace.io` converts in
both directions; you should never need to do this conversion by hand.

## `pipesight.meta` — one `TraceMeta`

```json
{
  "schema_version": 1,
  "source": "marker",
  "hostname": "ip-172-31-21-143",
  "gpu_name": null,
  "cpu_count_logical": 4,
  "cpu_count_physical": 2,
  "command": null,
  "wall_start_epoch_s": 1783920111.1880906,
  "perf_counter_offset_ns": 27497740817755
}
```

| Field | Meaning |
|---|---|
| `source` | `"marker"` (via `Profiler`), `"quicklook"` (via `pipesight run`), or `"torch_profiler"` (reserved for a future ingestion path — see the "Not yet supported" section below) |
| `gpu_name` | Set for quick-look traces (queried via NVML/`nvidia-smi`); `null` for marker traces, since a marker trace can span non-GPU work entirely |
| `cpu_count_logical` / `cpu_count_physical` | Captured at `Profiler`/quick-look start; `pipesight report`'s worker-count recommendation defaults to `cpu_count_physical` unless you pass `--physical-cores` |
| `command` | The full command line, for `source="quicklook"` traces only (`shlex.join`'d) |
| `wall_start_epoch_s` / `perf_counter_offset_ns` | See "Clock alignment" below |

## `pipesight.samples` — zero-touch mode's `Sample`s

Only populated for `source="quicklook"` traces (`pipesight run`). Each
entry is one poll of `psutil`/NVML at the configured `--interval`:

```json
{
  "ts_ns": 26845927344390,
  "cpu_percent": [0.0, 0.0, 0.0, 0.0],
  "gpu_util_pct": 88.0,
  "gpu_mem_used_mb": 2864.1875,
  "proc_id": null
}
```

`cpu_percent` is per logical CPU (`psutil.cpu_percent(percpu=True)`).
`gpu_util_pct`/`gpu_mem_used_mb` are `null` if no GPU was found or
`--no-gpu` was passed. `gpu_util_pct` is **device-wide, not
process-scoped** — see `pipesight.profiling.gpu.sample_gpu`'s docstring and
`examples/synthetic_pipeline/README.md` for what that means in practice on
a shared GPU. Marker traces (`source="marker"`) have an empty `samples`
list — spans already give exact per-stage timing, so there's nothing
coarse polling would add.

## Clock alignment (why `wall_start_epoch_s` / `perf_counter_offset_ns` exist)

Every timestamp in a `Span`/`Sample` (`start_ns`, `end_ns`, `ts_ns`) is a
raw `time.perf_counter_ns()` value — monotonic and cheap, but **only
comparable within the process that produced it**. Two different processes'
`perf_counter_ns()` clocks have no defined relationship to each other.

That matters because `pipesight report --merge-dir` combines trace files
from multiple worker processes (e.g. one per `ProcessPoolExecutor` worker
in a batch pipeline). `trace.io.merge()` handles this by converting every
timestamp to wall-clock nanoseconds first:

```
epoch_ns(meta, t_ns) = round(meta.wall_start_epoch_s * 1e9) + (t_ns - meta.perf_counter_offset_ns)
```

`wall_start_epoch_s` (`time.time()`) and `perf_counter_offset_ns`
(`time.perf_counter_ns()`) are captured together, at the same instant, when
a `Profiler` or quick-look sampler starts — so this formula re-expresses
any timestamp from that process as "nanoseconds since that process
started," in a common, cross-process-comparable unit. `merge()` then
re-bases the whole merged trace to start at zero. You never need to do this
by hand; it's only worth understanding if you're debugging why a merged
trace's ordering looks off (check that every input trace file actually has
correct, non-default `wall_start_epoch_s`/`perf_counter_offset_ns` values —
both are set automatically by `Profiler.__init__`/`run_quicklook`, so this
only bites if a trace was hand-constructed).

## Not yet supported

- **Ingesting a `torch.profiler` trace.** `TraceSource` already reserves
  `"torch_profiler"` for this, and the Chrome Trace Format choice makes it
  cheap in principle (re-tag its events as `Span`s and merge them in), but
  there's no `analysis/torch_ingest.py` yet — this is deferred, tracked
  in the project plan's Phase 6.
- **`nsys` kernel-level traces.** Similarly deferred; `pipesight run --nsys`
  doesn't exist yet.
