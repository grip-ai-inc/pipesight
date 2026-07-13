"""Synthetic decode -> gpu_compute -> postprocess stages, shaped after the
real grip-ai DROID-SLAM (GPU) + MediaPipe hand-pose/retarget/upload (CPU)
diagnosis this whole tool generalizes: a CPU stage, a real GPU stage, and a
second CPU stage sized to dominate wall time -- so the GPU sits idle during
postprocess in the naive sequential loop, exactly like the real pipeline did
before its Pipeline retrofit.

Durations are shaped to roughly: decode ~50ms, gpu_compute ~150-250ms (real
CUDA matmuls, torch.cuda.synchronize()'d), postprocess ~200ms -- postprocess
+ decode together (both CPU) exceed gpu_compute, so a naive sequential loop
leaves the GPU idle more often than not.
"""

from __future__ import annotations

import time

_MATMUL_SIZE = 1536
_MATMUL_ITERS = 70

_cuda_warmed_up = False


def warmup() -> None:
    """Pays CUDA context-init and cuBLAS algorithm-selection cost once,
    upfront, so it doesn't inflate item 0's gpu_compute measurement in
    either run_sequential.py or run_pipeline.py (both call this first, so
    the comparison between them stays apples-to-apples)."""
    global _cuda_warmed_up
    if _cuda_warmed_up:
        return
    _cuda_warmed_up = True
    gpu_compute(-1)


def decode_cpu(item_id: int) -> int:
    """~50ms CPU stage, standing in for frame decode/resize."""
    time.sleep(0.05)
    return item_id


def gpu_compute(item_id: int) -> int:
    """~150-250ms real GPU stage: a CUDA matmul loop, synchronized so the
    measured duration reflects actual device-busy time, standing in for
    DROID-SLAM tracking + bundle adjustment."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(_MATMUL_SIZE, _MATMUL_SIZE, device=device)
    for _ in range(_MATMUL_ITERS):
        x = x @ x
        x = x / x.abs().max()
    if device == "cuda":
        torch.cuda.synchronize()
    return item_id


def postprocess_cpu(item_id: int) -> dict:
    """~200ms CPU stage, standing in for hand-pose + retarget + upload --
    the stage that dominates wall time and leaves the GPU idle."""
    time.sleep(0.2)
    return {"item_id": item_id, "status": "done"}
