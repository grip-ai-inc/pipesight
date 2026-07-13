# synthetic_pipeline

A self-contained, runnable demo of the exact diagnosis pipesight generalizes:
a CPU `decode` stage, a real GPU `gpu_compute` stage (CUDA matmuls,
`torch.cuda.synchronize()`'d), and a CPU `postprocess` stage sized to
dominate wall time -- shaped after grip-ai's real DROID-SLAM (GPU) +
MediaPipe hand-pose/retarget/upload (CPU) pipeline. See `stages.py`.

Requires a CUDA GPU (falls back to CPU for `gpu_compute` if none is
available, which still works but won't show a meaningful "GPU idle" story).

## Three ways to see the story

Run from this directory (`examples/synthetic_pipeline/`):

### 1. Zero-touch (no code changes)

```bash
pipesight run --out quicklook_trace.json -- python run_sequential.py --n 8
pipesight report quicklook_trace.json
```

No stage names here (quick-look mode can't see inside the process), so
`report` only shows sample-based GPU idle %, not per-stage stats or
recommendations. **Caveat observed while validating this on a shared GPU**:
quick-look measures the *whole subprocess's* wall-clock lifetime, including
Python/torch import and CUDA context-init time (~2.7s of `torch`+`numpy`
import alone on the machine this was built on) -- which the script's own
internal timer doesn't count, since that only wraps the explicitly-timed
loop. And `nvidia-smi`/NVML utilization is device-wide, not
process-scoped -- on a GPU shared with other jobs, their usage shows up in
your idle-% too. Both are real, inherent limits of black-box process
wrapping, not bugs; see `pipesight.profiling.gpu.sample_gpu`'s docstring.
This is exactly why quick-look is the "first pass" tier, not the one you'd
act on -- legs 2 and 3 below are process-scoped and don't have this problem.

### 2. Marker-based profiling + recommendations

```bash
python run_sequential.py --n 8 --out sequential_trace.json
pipesight report sequential_trace.json --html sequential_timeline.html
```

Now `report` sees three named, `item_id`-correlated stages and should
report GPU idle around 60-65%, plus two recommendations: increasing
parallel workers, and overlapping `postprocess` with the next item's
`decode` -- pointing you at `Pipeline`.

### 3. Apply the fix and prove it

```bash
python run_pipeline.py --n 8 --out pipeline_trace.json
pipesight compare sequential_trace.json pipeline_trace.json --html diff.html
```

`run_pipeline.py` is the *same three stage functions*, just wired through
`Pipeline` instead of a sequential loop. On the machine this was built on:

```
                     sequential_trace.json pipeline_trace.json        delta
------------------------------------------------------------------------
wall-clock                        3.14s              1.53s       -51.3%
GPU idle %                        63.9%              16.7%       -47.2pp
```

A ≥20% wall-clock improvement and a material GPU-idle-% drop is the
regression-tested pass bar (see `tests/test_examples_end_to_end.py`,
marked `gpu`+`slow`) -- 51% and -47pp comfortably clears it.
