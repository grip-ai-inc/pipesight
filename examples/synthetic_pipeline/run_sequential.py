"""The "before" baseline: a naive sequential loop, exactly the shape the
real grip-ai preprocessing pipeline had -- decode, then GPU compute, then
postprocess, one item fully finished before the next one starts. Writes
sequential_trace.json for `pipesight report` / `pipesight compare`.
"""

from __future__ import annotations

import argparse
import time

from stages import decode_cpu, gpu_compute, postprocess_cpu, warmup

from pipesight import Profiler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of items to process")
    parser.add_argument("--out", default="sequential_trace.json")
    args = parser.parse_args()

    warmup()

    prof = Profiler(out_path=args.out)
    t0 = time.perf_counter()
    for i in range(args.n):
        with prof.stage("decode", device="cpu", item_id=i):
            frame = decode_cpu(i)
        with prof.stage("gpu_compute", device="gpu", item_id=i):
            frame = gpu_compute(frame)
        with prof.stage("postprocess", device="cpu", item_id=i):
            postprocess_cpu(frame)
    elapsed = time.perf_counter() - t0
    prof.close()

    print(f"sequential: {args.n} items in {elapsed:.2f}s, trace -> {args.out}")


if __name__ == "__main__":
    main()
