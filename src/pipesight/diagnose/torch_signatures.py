"""PyTorch-specific failure signatures.

Kept out of the generic `signatures` module on purpose: the vocabulary here
(`DataLoader`, `num_workers`, `persistent_workers`, the 'spawn'/'fork' start
methods, DataLoader's exact error strings) is PyTorch's, and pipesight's core
stays framework-agnostic. These are appended to the registry by
`signatures.default_signatures()`.

These signatures read *text* (a tail of stderr / a traceback), so they're the
pattern-matching tier -- strong hints from known messages, not measurements.
The OOM one additionally corroborates against the memory timeline when present.
"""

from __future__ import annotations

import re

from pipesight.diagnose.diagnose import DiagnoseContext, Diagnosis, Signature

# "DataLoader worker (pid 12345) is killed by signal: Killed."
# "DataLoader worker (pid(s) 12345, 12346) is killed by signal: Bus error."
_WORKER_KILLED = re.compile(
    r"DataLoader worker \(pid[^)]*\) is killed by signal:\s*([A-Za-z][A-Za-z ]*)",
    re.IGNORECASE,
)
# "DataLoader worker (pid(s) 12345) exited unexpectedly"
_WORKER_EXITED = re.compile(
    r"DataLoader worker \(pid[^)]*\) exited unexpectedly", re.IGNORECASE
)
_CUDA_FORK = re.compile(
    r"Cannot re-initialize CUDA in forked subprocess|"
    r"must use the ['\"]spawn['\"] start method",
    re.IGNORECASE,
)
_PICKLE = re.compile(
    r"can'?t pickle|cannot pickle|PicklingError|"
    r"Can't get local object|Can't pickle local object",
    re.IGNORECASE,
)
_TOO_MANY_FDS = re.compile(
    r"too many open files|received 0 items of ancdata|\[Errno 24\]",
    re.IGNORECASE,
)


def torch_dataloader_worker_killed(ctx: DiagnoseContext) -> Diagnosis | None:
    """The headline case: `DataLoader worker (pid N) is killed by signal: X`.

    The signal name in the message tells us which failure family it is:
    'Killed' -> OOM, 'Bus error' -> shared-memory exhaustion.
    """
    m = _WORKER_KILLED.search(ctx.stderr)
    if not m:
        return None
    signal_name = m.group(1).strip()
    lowered = signal_name.lower()

    mem = ctx.memory
    mem_line = ""
    evidence = {"worker_signal": signal_name}
    if mem is not None and mem.has_data and mem.peak_used_pct is not None:
        evidence["peak_mem_pct"] = round(mem.peak_used_pct, 1)
        evidence["peak_proc_count"] = mem.peak_proc_count
        if mem.near_ceiling:
            mem_line = (
                f" pipesight's own memory sampling corroborates this: host RAM peaked at "
                f"{mem.peak_used_pct:.0f}% of {(mem.sys_total_mb or 0) / 1024:.1f} GiB"
                + (
                    f" across up to {mem.peak_proc_count} processes"
                    if mem.peak_proc_count
                    else ""
                )
                + "."
            )
        else:
            mem_line = (
                f" Note: sampled host RAM only peaked at {mem.peak_used_pct:.0f}%, below the "
                f"ceiling -- if this was OOM, the spike may have been faster than the sampling "
                f"interval caught, or confined to a single worker."
            )

    if "bus" in lowered:  # Bus error -> shared memory, not host RAM
        return Diagnosis(
            signature_id="torch_dataloader_worker_killed_bus",
            category="shm",
            title="DataLoader worker killed by SIGBUS -- shared-memory exhaustion",
            confidence="high",
            what_happened=(
                f"A DataLoader worker was killed by signal '{signal_name}'. SIGBUS from a "
                f"DataLoader worker almost always means /dev/shm (shared memory used to pass "
                f"tensors from workers to the main process) ran out.{mem_line}"
            ),
            likely_causes=[
                "/dev/shm too small (Docker default is 64MB) for the tensors being passed.",
                "High num_workers * prefetch_factor keeping many large batches in shm at once.",
            ],
            suggested_fixes=[
                "Increase shared memory: `docker run --shm-size=1g` / `--ipc=host`.",
                "Lower num_workers and/or prefetch_factor.",
                "Use `torch.multiprocessing.set_sharing_strategy('file_system')` to spill "
                "sharing to files instead of /dev/shm (watch your open-file limit then).",
            ],
            evidence=evidence,
        )

    # "Killed" / SIGKILL -> OOM on host RAM.
    confidence = "high"
    return Diagnosis(
        signature_id="torch_dataloader_worker_killed_oom",
        category="oom",
        title="DataLoader worker killed by the OOM killer (host RAM exhausted)",
        confidence=confidence,
        what_happened=(
            f"A DataLoader worker was killed by signal '{signal_name}'. That signal on a worker "
            f"is the Linux OOM killer reclaiming host RAM -- the main process then re-raises it "
            f"as the RuntimeError you see.{mem_line}"
        ),
        likely_causes=[
            "num_workers too high: each worker is a full process holding its own copy of "
            "dataset state / buffers, and total RSS overran host RAM.",
            "A large per-sample memory footprint (big decoded images/tensors) times "
            "prefetch_factor times num_workers.",
            "persistent_workers=True keeping worker memory resident across epochs.",
            "A leak: worker RSS climbing every iteration (caching, accumulating lists).",
        ],
        suggested_fixes=[
            "Lower num_workers first -- it's the biggest lever on total dataloader RAM.",
            "Lower prefetch_factor (default 2) to keep fewer batches buffered per worker.",
            "Shrink per-sample memory: decode lazily, downscale earlier, avoid holding whole "
            "arrays; make sure the Dataset isn't caching everything in __init__.",
            "If RSS climbs monotonically it's a leak -- fix the accumulation rather than just "
            "cutting workers.",
            "As a diagnostic, set num_workers=0: if the OOM disappears, it's worker fan-out; "
            "if it persists, the main process itself is the memory hog.",
        ],
        evidence=evidence,
    )


def torch_worker_exited(ctx: DiagnoseContext) -> Diagnosis | None:
    if not _WORKER_EXITED.search(ctx.stderr) or _WORKER_KILLED.search(ctx.stderr):
        # If the more specific "killed by signal" form matched, defer to it.
        return None
    return Diagnosis(
        signature_id="torch_dataloader_worker_exited",
        category="oom",
        title="DataLoader worker exited unexpectedly",
        confidence="medium",
        what_happened=(
            "A DataLoader worker process died without a signal message. Most often this is "
            "still an OOM kill (whose signal line got lost) or an unhandled exception inside "
            "the worker's dataset code."
        ),
        likely_causes=[
            "Out-of-memory kill of the worker (see the OOM guidance).",
            "An exception raised inside Dataset.__getitem__ / collate_fn in the worker.",
        ],
        suggested_fixes=[
            "Re-run with num_workers=0 so worker exceptions surface with a full traceback in "
            "the main process instead of a terse 'exited unexpectedly'.",
            "If it only happens with workers > 0 and memory is tight, treat it as OOM and cut "
            "num_workers / prefetch_factor.",
        ],
        evidence={},
    )


def torch_cuda_fork(ctx: DiagnoseContext) -> Diagnosis | None:
    if not _CUDA_FORK.search(ctx.stderr):
        return None
    return Diagnosis(
        signature_id="torch_cuda_fork",
        category="fork_cuda",
        title="CUDA re-initialized in a forked worker -- wrong multiprocessing start method",
        confidence="high",
        what_happened=(
            "A worker tried to use CUDA after being created with the 'fork' start method. A "
            "forked child inherits a CUDA context that can't be reused, so torch refuses and "
            "tells you to use 'spawn'."
        ),
        likely_causes=[
            "CUDA was initialized in the parent (e.g. a `.cuda()` / `torch.cuda.*` call, or a "
            "model moved to GPU) before workers were forked.",
            "The default start method is 'fork' on Linux, which is incompatible with "
            "already-initialized CUDA.",
        ],
        suggested_fixes=[
            "Set the spawn start method at program start: "
            "`torch.multiprocessing.set_start_method('spawn', force=True)` (guard it under "
            "`if __name__ == '__main__':`).",
            "Or avoid touching CUDA before the DataLoader/workers start -- keep tensor "
            "construction in workers on CPU and move to GPU in the main loop.",
            "Note spawn re-imports your module in each worker, so heavy import-time side "
            "effects must be guarded by `if __name__ == '__main__':`.",
        ],
        evidence={},
    )


def torch_unpicklable_spawn(ctx: DiagnoseContext) -> Diagnosis | None:
    if not _PICKLE.search(ctx.stderr):
        return None
    return Diagnosis(
        signature_id="torch_unpicklable_spawn",
        category="pickle",
        title="Unpicklable object crossing into a worker (spawn start method)",
        confidence="medium",
        what_happened=(
            "Something couldn't be pickled to hand off to a worker process. With the 'spawn' "
            "start method, the dataset and its arguments are pickled and sent to each worker; "
            "lambdas, local closures, open file handles, and lock objects can't be pickled."
        ),
        likely_causes=[
            "A lambda or locally-defined function/closure used as transform / collate_fn / "
            "worker_init_fn.",
            "The Dataset holding an unpicklable member (an open file, a DB connection, a lock, "
            "a generator).",
        ],
        suggested_fixes=[
            "Replace lambdas/local closures with module-level functions or picklable callables.",
            "Open files / DB connections lazily inside the worker (in __getitem__ or "
            "worker_init_fn), not in __init__.",
            "If you don't need spawn, forking avoids the pickle step -- but fork is "
            "incompatible with pre-initialized CUDA (see the fork/CUDA guidance).",
        ],
        evidence={},
    )


def torch_too_many_fds(ctx: DiagnoseContext) -> Diagnosis | None:
    if not _TOO_MANY_FDS.search(ctx.stderr):
        return None
    return Diagnosis(
        signature_id="torch_too_many_open_files",
        category="fds",
        title="Too many open files -- file-descriptor sharing strategy exhausted ulimit",
        confidence="high",
        what_happened=(
            "The process ran out of file descriptors (or saw 'received 0 items of ancdata'). "
            "PyTorch's default 'file_descriptor' sharing strategy opens an fd per shared tensor; "
            "with many workers and buffered batches this can blow past the ulimit."
        ),
        likely_causes=[
            "Default 'file_descriptor' sharing strategy + many workers * prefetched batches.",
            "A low `ulimit -n` (open-files limit).",
        ],
        suggested_fixes=[
            "Switch strategy: `torch.multiprocessing.set_sharing_strategy('file_system')`.",
            "Raise the limit: `ulimit -n 65535` (or the container/systemd equivalent).",
            "Reduce num_workers / prefetch_factor so fewer tensors are shared at once.",
        ],
        evidence={},
    )


def torch_signatures() -> list[Signature]:
    return [
        torch_dataloader_worker_killed,
        torch_worker_exited,
        torch_cuda_fork,
        torch_unpicklable_spawn,
        torch_too_many_fds,
    ]
