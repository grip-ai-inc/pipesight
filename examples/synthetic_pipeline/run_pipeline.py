"""The "after": the exact same three stages, wired through pipesight's
Pipeline instead of a sequential loop -- item N+1's decode/gpu_compute can
start while item N's postprocess is still running. This is the retrofit;
compare its trace against sequential_trace.json with `pipesight compare`.
"""

from __future__ import annotations

import argparse
import time

from stages import decode_cpu, gpu_compute, postprocess_cpu, warmup

from pipesight import Pipeline, Profiler, StageSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of items to process")
    parser.add_argument("--out", default="pipeline_trace.json")
    args = parser.parse_args()

    warmup()

    stages = [
        StageSpec("decode", decode_cpu, device="cpu", workers=2),
        StageSpec("gpu_compute", gpu_compute, device="gpu", workers=1),
        StageSpec("postprocess", postprocess_cpu, device="cpu", workers=2),
    ]

    with Profiler(out_path=args.out) as prof:
        t0 = time.perf_counter()
        with Pipeline(stages, profiler=prof) as pipeline:
            results = list(pipeline.run(range(args.n)))
        elapsed = time.perf_counter() - t0

    assert len(results) == args.n
    print(f"pipelined: {args.n} items in {elapsed:.2f}s, trace -> {args.out}")


if __name__ == "__main__":
    main()
