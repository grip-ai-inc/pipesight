# Worked example: retrofitting a real sequential loop

This walks through applying pipesight to a real pipeline: grip-ai's Ego4D
preprocessing script, the actual motivating case for this whole tool (see
the root `README.md`'s "Why"). It's a worked example only — none of this
has been applied to grip-ai itself; the point is to show the process on
real, unmodified code you can go read yourself, not a toy.

## The starting point

`scripts/preprocess_ego4d.py`'s `process_segment()` does, per segment: build
the clip, run DROID-SLAM (GPU), run MediaPipe hand-pose (CPU), lift/retarget
the hand keypoints (CPU), and return a result blob. Its caller,
`process_video()` in `scripts/preprocess_ego4d_streaming.py`, loops over
every segment in a video and uploads each result to S3 afterward:

```python
# scripts/preprocess_ego4d_streaming.py (excerpt, unmodified)
for seg in segments:
    try:
        blob = process_segment(seg, video_root_for_loader, slam, hand, retarget, target_fps)
    except Exception as e:
        logger.warning("skip %s %.2f-%.2f: %s", seg.video_uid, seg.start_sec, seg.end_sec, e)
        continue

    with tempfile.TemporaryDirectory(dir=str(scratch_dir)) as tmpdir:
        # ... serialize blob to .npz/.meta.json, upload() to S3 ...
```

and inside `process_segment()` (`scripts/preprocess_ego4d.py`):

```python
def process_segment(seg, video_root, slam, hand, retarget, target_fps) -> dict:
    clip = build_clip_from_segment(seg, video_root, target_fps=target_fps)   # CPU
    images = np.stack([f.image for f in clip.frames])

    slam_res = slam.run(images, fps=target_fps)                              # GPU
    hp_res = hand.run(images)                                                # CPU

    # ... wrist lift, optional retarget.retarget(kps_cam) ...                # CPU
    return {...}
```

This is exactly the "sequential loop over items, mixing CPU and GPU
stages" shape pipesight targets: `decode` (build the clip), `slam` (GPU),
`hand_pose` + `retarget` (CPU), `upload` (CPU/network), one segment fully
finished before the next one starts.

## Step 1 — instrument first, change nothing

Before touching the loop's structure, get a baseline trace. Wrap each
stage in `Profiler.stage(...)`, tagging every span with the same
`item_id` so pipesight can correlate stages across segments:

```python
from pipesight import Profiler

prof = Profiler(out_path=f"trace.worker_{os.getpid()}.json")   # one per worker process

def process_segment(seg, video_root, slam, hand, retarget, target_fps) -> dict:
    item_id = seg.video_uid  # or f"{seg.video_uid}:{seg.start_sec}-{seg.end_sec}" if segments overlap in time

    with prof.stage("decode", device="cpu", item_id=item_id):
        clip = build_clip_from_segment(seg, video_root, target_fps=target_fps)
        images = np.stack([f.image for f in clip.frames])

    with prof.stage("slam", device="gpu", item_id=item_id):
        slam_res = slam.run(images, fps=target_fps)

    with prof.stage("hand_pose", device="cpu", item_id=item_id):
        hp_res = hand.run(images)

    with prof.stage("retarget", device="cpu", item_id=item_id):
        # ... wrist lift, retarget.retarget(kps_cam) ...
        ...
    return {...}
```

...and around the upload call in `process_video()`:

```python
with prof.stage("upload", device="cpu", item_id=seg.video_uid):
    upload(npz_path, out_key, ...)
    upload(meta_path, meta_key, ...)
```

Register `atexit.register(prof.close)` next to the existing
`atexit.register(hand.close)` in `_init_worker` (see
`preprocess_ego4d_streaming.py`) so each `ProcessPoolExecutor` worker
flushes its own trace file on exit. Run a real (or `--limit-videos 1`)
batch, then combine every worker's trace and look at it:

```bash
pipesight report --merge-dir /tmp/egoscale_scratch/traces/ --html before.html
```

This is purely additive — nothing about the pipeline's behavior or output
changes, only what you can see about its timing.

## Step 2 — read the report

Expect something like the pattern already measured manually for this exact
pipeline (see the root README's "Why", and the conversation that produced
this tool): GPU idle a majority of the time, because `hand_pose` +
`retarget` + `upload` are CPU-only and run after `slam` finishes, not
alongside it. `pipesight report` should surface two recommendation kinds:

- **`increase_workers`** — bump `--num-workers` up to the box's *physical*
  core count (the machine used to build this had 2 physical / 4 logical
  cores; `--num-workers` was already documented in
  `preprocess_ego4d_streaming.py` to matter for exactly this reason).
- **`pipeline_overlap`** — `retarget`/`upload` (whichever finishes a
  segment) finishing strictly before the next segment's `decode` starts, in
  most consecutive segment pairs. This is the one `Pipeline` fixes.

## Step 3 — apply the overlap fix with `Pipeline`

Convert the segment loop into stages, respecting two real constraints this
pipeline actually has:

```python
from pipesight.pipeline import Pipeline, StageSpec

def decode_fn(seg):
    clip = build_clip_from_segment(seg, video_root_for_loader, target_fps=target_fps)
    return seg, np.stack([f.image for f in clip.frames])

def slam_fn(pair):
    seg, images = pair
    return seg, images, slam.run(images, fps=target_fps)

def hand_and_retarget_fn(triple):
    seg, images, slam_res = triple
    hp_res = hand.run(images)
    # ... wrist lift, retarget.retarget(kps_cam) ...
    return build_result_blob(seg, slam_res, hp_res, ...)

def upload_fn(blob):
    # ... serialize to .npz/.meta.json, upload() to S3 ...
    return blob["clip_id"]

stages = [
    StageSpec("decode", decode_fn, device="cpu", workers=2),
    # DROID-SLAM holds one physical GPU's worth of CUDA context; see
    # SlamRunner's per-video Droid() lifecycle (src/perception/slam.py) --
    # concurrency beyond 1 here would mean multiple Droid() instances
    # fighting over the same GPU, not real parallelism.
    StageSpec("slam", slam_fn, device="gpu", workers=1),
    # MUST stay workers=1: mediapipe's VIDEO-mode HandLandmarker is a
    # single long-lived instance reused across every segment, and requires
    # strictly increasing timestamps across its whole lifetime (see
    # HandPoseEstimator.__init__'s comment and _next_ts_ms in
    # src/perception/hand_pose.py) -- concurrent calls would race on that
    # counter and violate the ordering MediaPipe expects.
    StageSpec("hand_and_retarget", hand_and_retarget_fn, device="cpu", workers=1),
    # Network-bound, not CPU-bound -- fans out fine.
    StageSpec("upload", upload_fn, device="cpu", workers=4),
]

with Pipeline(stages, profiler=prof, on_error="skip") as pipeline:
    for clip_id in pipeline.run(segments):
        logger.info("done: %s", clip_id)
```

Two judgment calls worth calling out:

- **`on_error="skip"`** directly generalizes the existing
  `try: ... except Exception: logger.warning(...); continue` pattern
  already in `process_video()` — one bad segment shouldn't stop the batch.
  `Pipeline` logs the same kind of warning (via `PipelineItemError`) and
  moves on.
- **`ordered=True` (the default) vs `False`**: the current code doesn't
  care what order segments finish in — each uploads to its own S3 key
  independently. `ordered=False` would let `Pipeline` yield whichever
  segment finishes first rather than holding faster segments back behind a
  slower one still in flight, which is very likely a small extra
  throughput win here. Kept as `ordered=True` above only because it's the
  safer default to reason about first; switching is a one-line change
  once you trust the retrofit.

## Step 4 — prove it

```bash
pipesight compare before.json after.json --html diff.html
```

Same pass bar as `examples/synthetic_pipeline/` (see its README and
`tests/test_examples_end_to_end.py`): expect wall-clock time down and GPU
idle % down. If `hand_and_retarget`'s single-worker constraint caps how
much the overlap fix can buy you, `recommend_worker_count`'s suggestion
(more `--num-workers`, i.e. more whole *processes* each running this same
`Pipeline`) is the complementary lever — the two compose, they aren't
alternatives (see the root README's "Why" and `Pipeline`'s docstring in
`src/pipesight/pipeline/pipeline.py`).
