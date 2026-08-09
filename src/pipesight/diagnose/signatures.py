"""The signature playbook: framework-agnostic failure signatures, plus a
registry that folds in the optional PyTorch-specific ones.

Signatures are just functions `DiagnoseContext -> Diagnosis | None`. Keeping the
generic set here (signals, host OOM, shared-memory bus errors, file-descriptor
exhaustion) separate from `torch_signatures` preserves pipesight's
framework-agnostic core -- the PyTorch vocabulary (`num_workers`,
`persistent_workers`, DataLoader's own error strings) lives in its own module
and is simply appended to the registry.
"""

from __future__ import annotations

import re

from pipesight.diagnose.diagnose import DiagnoseContext, Diagnosis, Signature

# Signal numbers we recognize by name in explanations (POSIX).
SIGKILL, SIGBUS, SIGSEGV = 9, 7, 11

_OOM_TEXT = re.compile(
    r"out of memory|oom-kill|killed process \d+|cgroup out of memory|"
    r"\bkilled\b\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _memory_evidence(ctx: DiagnoseContext) -> dict:
    mem = ctx.memory
    if mem is None or not mem.has_data or mem.peak_used_pct is None:
        return {}
    return {
        "peak_mem_pct": round(mem.peak_used_pct, 1),
        "sys_total_mb": round(mem.sys_total_mb or 0, 1),
        "peak_proc_rss_mb": round(mem.peak_proc_rss_mb, 1) if mem.peak_proc_rss_mb else None,
        "peak_proc_count": mem.peak_proc_count,
    }


def host_oom(ctx: DiagnoseContext) -> Diagnosis | None:
    """The launched process itself was OOM-killed (SIGKILL / dmesg OOM text).

    Distinct from a *worker* being killed (that's the torch DataLoader
    signature) -- this is the main process or a single-process run dying.
    """
    signalled = ctx.term_signal == SIGKILL
    text_hit = bool(_OOM_TEXT.search(ctx.stderr))
    if not (signalled or text_hit):
        return None

    mem = ctx.memory
    near = mem is not None and mem.near_ceiling
    # An explicit SIGKILL with RAM already at the ceiling is about as close to
    # certain as post-mortem inference gets; a bare SIGKILL could also be a
    # manual `kill -9`, so we hedge to medium without the memory corroboration.
    confidence: str = "high" if near else "medium"

    evidence = _memory_evidence(ctx)
    evidence["term_signal"] = ctx.term_signal
    detail = "The process was killed by SIGKILL (signal 9)." if signalled else (
        "stderr/kernel log shows an out-of-memory kill."
    )
    if mem is not None and mem.peak_used_pct is not None:
        detail += (
            f" Peak host RAM during the run reached {mem.peak_used_pct:.0f}% of "
            f"{(mem.sys_total_mb or 0) / 1024:.1f} GiB"
            + (" -- at the ceiling." if near else ".")
        )

    return Diagnosis(
        signature_id="host_oom",
        category="oom",
        title="Process killed by the OS out-of-memory killer",
        confidence=confidence,
        what_happened=detail,
        likely_causes=[
            "Host RAM (not GPU memory) was exhausted; the kernel SIGKILLed the process.",
            "Too many parallel workers, each holding a large in-memory share of data.",
            "A growing cache / unbounded prefetch buffer / a leak accumulating over iterations.",
            "In a container: a cgroup memory limit lower than the visible host RAM.",
        ],
        suggested_fixes=[
            "Reduce parallelism (fewer worker processes / lower num_workers).",
            "Shrink per-worker footprint: smaller prefetch/queue depth, lazy-load samples, "
            "avoid materializing whole datasets in RAM.",
            "Watch RSS over time (pipesight already samples it) to tell a leak (monotonic climb) "
            "from a legitimately-too-large working set (high but flat).",
            "In containers, raise the memory limit or lower worker count to fit the cgroup cap.",
        ],
        evidence=evidence,
    )


def bus_error_shm(ctx: DiagnoseContext) -> Diagnosis | None:
    """SIGBUS / shared-memory exhaustion -- classically /dev/shm too small."""
    signalled = ctx.term_signal == SIGBUS
    text_hit = bool(
        re.search(
            r"bus error|unable to open shared memory object|"
            r"/dev/shm|no space left on device",
            ctx.stderr,
            re.IGNORECASE,
        )
    )
    if not (signalled or text_hit):
        return None
    return Diagnosis(
        signature_id="bus_error_shm",
        category="shm",
        title="Bus error / shared-memory (/dev/shm) exhaustion",
        confidence="high" if (signalled and text_hit) else "medium",
        what_happened=(
            "The process hit SIGBUS or failed to open a shared-memory segment. Worker "
            "processes pass tensors/arrays through shared memory (/dev/shm); when that "
            "fills up, mapping fails with a bus error."
        ),
        likely_causes=[
            "/dev/shm is too small -- very common in Docker (default 64MB) and some clusters.",
            "Many workers each holding large shared-memory buffers at once.",
        ],
        suggested_fixes=[
            "Raise shared memory: `docker run --shm-size=1g` (or `--ipc=host`); on k8s mount a "
            "larger `emptyDir` with `medium: Memory` at /dev/shm.",
            "Reduce num_workers and/or prefetch_factor so fewer buffers coexist.",
            "For PyTorch specifically, switch the sharing strategy to file-backed: "
            "`torch.multiprocessing.set_sharing_strategy('file_system')`.",
        ],
        evidence={"term_signal": ctx.term_signal},
    )


def segfault(ctx: DiagnoseContext) -> Diagnosis | None:
    if ctx.term_signal != SIGSEGV:
        return None
    return Diagnosis(
        signature_id="segfault",
        category="segfault",
        title="Segmentation fault (SIGSEGV) in the worker/process",
        confidence="low",
        what_happened="The process crashed with SIGSEGV -- a native-code memory fault.",
        likely_causes=[
            "A native extension (a C/CUDA op, a codec, a compiled data-loading lib) faulted.",
            "A version mismatch between compiled libraries (e.g. torch vs a custom op).",
            "fork() after a non-fork-safe native library initialized threads/handles.",
        ],
        suggested_fixes=[
            "Re-run with a single worker (num_workers=0) to see if the fault is in the "
            "worker/parallel path.",
            "Get a native traceback with `faulthandler` (`python -X faulthandler`) or gdb.",
            "Check for ABI/version mismatches between torch and any compiled extensions.",
        ],
        evidence={"term_signal": ctx.term_signal},
    )


def generic_nonzero_exit(ctx: DiagnoseContext) -> Diagnosis | None:
    """Last-resort catch-all: the run failed but no specific signature matched."""
    if not ctx.failed:
        return None
    return Diagnosis(
        signature_id="generic_nonzero_exit",
        category="unknown",
        title="Run failed, but no specific failure signature matched",
        confidence="low",
        what_happened=(
            "The command ended abnormally "
            + (
                f"(killed by signal {ctx.term_signal})."
                if ctx.term_signal is not None
                else f"(exit code {ctx.exit_code})."
            )
            + " No known dataloader/OOM/shm signature matched the captured output."
        ),
        likely_causes=["An application error, or a failure mode not yet in the playbook."],
        suggested_fixes=[
            "Read the full traceback (only a tail of stderr is captured here).",
            "Re-run with more logging; if it's dataloader-related, try num_workers=0 to isolate "
            "the parallel path from the data logic itself.",
        ],
        evidence={"exit_code": ctx.exit_code, "term_signal": ctx.term_signal},
    )


def default_signatures() -> list[Signature]:
    """Full registry: PyTorch-specific signatures first (most specific), then
    the generic fallbacks. `diagnose()` keeps the most confident per category,
    so ordering only breaks confidence ties in favor of the specific ones."""
    from pipesight.diagnose.torch_signatures import torch_signatures

    return [
        *torch_signatures(),
        host_oom,
        bus_error_shm,
        segfault,
        generic_nonzero_exit,
    ]
