# pipesight

Profile the CPU/GPU stages of an end-to-end pipeline, find out where the GPU
is sitting idle, get ranked recommendations for fixing it, and apply the fix
with a drop-in `Pipeline`/`Stage` runtime instead of hand-rolling threading
code.

## Why

Mixed CPU/GPU pipelines (decode → GPU inference → postprocess → upload, ...)
are usually built sequentially first and never revisited for utilization.
The GPU sits idle during every CPU-only stage, and the fix — overlapping
stage N+1's CPU work with stage N's GPU work — is easy to describe and
annoying to hand-implement correctly (ordering, backpressure, error
handling, clean shutdown). pipesight profiles the pipeline, tells you
exactly which stages don't overlap but could, and gives you the primitive to
fix it in a few lines.

## Install

```bash
pip install -e ".[dev,gpu]"   # gpu extra: NVML-based utilization sampling
```

## Quick start

**Zero-touch** (no code changes — coarse CPU/GPU utilization over wall time):

```bash
pipesight run --out trace.json -- python my_pipeline.py
pipesight report trace.json
```

**Precise, stage-named profiling** (needed for overlap-opportunity detection
and for driving `Pipeline`):

```python
from pipesight import Profiler

prof = Profiler(out_path="trace.json")
for item in items:
    with prof.stage("decode", device="cpu", item_id=item.id):
        frame = decode(item)
    with prof.stage("infer", device="gpu", item_id=item.id):
        result = model(frame)
prof.close()
```

```bash
pipesight report trace.json --html timeline.html
```

**Applying an overlap recommendation** with `Pipeline`:

```python
from pipesight import Profiler
from pipesight.pipeline import Pipeline, StageSpec

stages = [
    StageSpec("decode", decode_fn, device="cpu", workers=2),
    StageSpec("infer",  infer_fn,  device="gpu", workers=1),
    StageSpec("upload", upload_fn, device="cpu", workers=4),
]
with Pipeline(stages, profiler=Profiler(out_path="pipeline_trace.json")) as pipeline:
    for result in pipeline.run(items):
        handle(result)
```

```bash
pipesight compare trace.json pipeline_trace.json --html diff.html
```

See `examples/synthetic_pipeline/` for a runnable, self-contained
before/after demo, and `docs/pipeline_retrofit.md` for a worked example of
retrofitting a real sequential loop.

## Layout

- `pipesight.trace` — the `Span`/`Sample`/`Trace` data model and Chrome
  Trace Event Format JSON codec (`save`/`load`/`merge`). Traces open
  directly in `chrome://tracing` / `ui.perfetto.dev`.
- `pipesight.profiling` — `Profiler` (opt-in stage markers) and the
  zero-touch quick-look sampler.
- `pipesight.analysis` — per-stage stats, GPU idle-gap detection,
  cross-iteration overlap-opportunity detection, ranked recommendations.
- `pipesight.pipeline` — `Pipeline`/`StageSpec`, the overlap runtime.
- `pipesight.viz` — self-contained HTML timeline rendering.
- `pipesight.cli` — `pipesight run|report|compare|version`.

## License

MIT
