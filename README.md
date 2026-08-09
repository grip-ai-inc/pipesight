# pipesight

Profile the CPU/GPU stages of an end-to-end pipeline, find out where the GPU is sitting idle, get ranked recommendations for fixing it, and apply the fix with a drop-in `Pipeline`/`Stage` runtime instead of hand-rolling threading code.

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

Note: GPU utilization here comes from `nvidia-smi`/NVML, which is
device-wide, not process-scoped -- on a shared GPU this includes *every*
process's usage, not just your command's. Treat zero-touch idle-% as noisy
on shared hardware; the marker API below is scoped to your own process and
is the more reliable tier for anything you're going to act on.

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

## Diagnosing failures (dataloader crashes, OOM, fork/spawn)

Profiling assumes the run *finishes*. When it instead dies with something like
`DataLoader worker (pid N) is killed by signal: Killed`, use `diagnose`:

```bash
pipesight diagnose -- python train.py     # run it (tees stderr) and explain the failure
pipesight diagnose run_trace.json         # diagnose a trace a previous run saved
pipesight diagnose --log train.log        # pattern-match an existing log file
```

`diagnose` matches the run's failure signals — terminating signal, exit code,
and a tail of stderr — against a playbook of known failure signatures (host
OOM, `/dev/shm`/SIGBUS shared-memory exhaustion, `fork`-vs-`spawn` + CUDA,
unpicklable datasets under spawn, file-descriptor exhaustion) and prints ranked,
advisory explanations with concrete fixes (lower `num_workers`/`prefetch_factor`,
raise `--shm-size`, switch the multiprocessing start method, …).

Crucially, the live `-- <command>` form also samples **host memory** (system
RAM plus the target's whole process-tree RSS) throughout the run, so an OOM kill
is *corroborated* against a real memory timeline rather than guessed — and the
same memory signal keeps the profiler's worker-count recommendation honest
("you're CPU-bound and could add workers, but host RAM is near the ceiling").
`pipesight run`/`report` surface this as a `== Memory ==` section.

Everything `diagnose` prints is advisory: a signature matching is strong
evidence from a known pattern, not proof.

## CLI reference

```
$ pipesight --help
usage: pipesight [-h] [--version] {run,report,compare,diagnose,version} ...

positional arguments:
  {run,report,compare,diagnose,version}
    run                 Profile an arbitrary command with no code changes
                        (zero-touch)
    report              Analyze a captured trace and print recommendations
    compare             Compare two traces (e.g. before/after a Pipeline
                        retrofit)
    diagnose            Explain why a pipeline run failed
                        (dataloader/OOM/shm/fork triage)
    version             Print the pipesight version

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

```
$ pipesight run --help
usage: pipesight run [-h] [--interval INTERVAL] [--out OUT] [--no-gpu] -- <command...>

Profile an arbitrary command with no code changes (zero-touch).

Everything after a literal `--` is run as the target command, e.g.:
    pipesight run --out trace.json -- python my_script.py --arg1 val1

options:
  -h, --help           show this help message and exit
  --interval INTERVAL  Sampling interval in seconds
  --out OUT            Output trace JSON path
  --no-gpu             Skip GPU utilization sampling
```

`pipesight report --help`, `pipesight compare --help`, and `pipesight diagnose
--help` follow the same pattern — run them directly for the full flag list
(`--merge-dir`, `--html`, `--physical-cores` on `report`; `--html` on `compare`;
`--log`, `--trace`, `--out` on `diagnose`).

## Layout

- `pipesight.trace` — the `Span`/`Sample`/`Trace` data model and Chrome
  Trace Event Format JSON codec (`save`/`load`/`merge`). Traces open
  directly in `chrome://tracing` / `ui.perfetto.dev`.
- `pipesight.profiling` — `Profiler` (opt-in stage markers) and the
  zero-touch quick-look sampler (CPU/GPU utilization + host/process-tree
  memory).
- `pipesight.analysis` — per-stage stats, GPU idle-gap detection,
  cross-iteration overlap-opportunity detection, host-memory pressure
  analysis, ranked recommendations.
- `pipesight.diagnose` — failure-signature playbook (generic + optional
  PyTorch-specific) turning a crashed run's signals into ranked, advisory
  explanations.
- `pipesight.pipeline` — `Pipeline`/`StageSpec`, the overlap runtime.
- `pipesight.viz` — self-contained HTML timeline rendering.
- `pipesight.cli` — `pipesight run|report|compare|diagnose|version`.

## License

MIT
