from __future__ import annotations

from pipesight.analysis.memory import memory_from_samples
from pipesight.diagnose.diagnose import DiagnoseContext, diagnose
from pipesight.trace.schema import Sample


def _ctx(stderr="", exit_code=None, term_signal=None, memory=None):
    return DiagnoseContext(
        stderr=stderr, exit_code=exit_code, term_signal=term_signal, memory=memory
    )


def _mem(used, total, rss=None, count=None):
    return memory_from_samples(
        [
            Sample(
                ts_ns=0,
                cpu_percent=[0.0],
                sys_mem_used_mb=used,
                sys_mem_total_mb=total,
                proc_rss_mb=rss,
                proc_count=count,
            )
        ]
    )


WORKER_KILLED = "RuntimeError: DataLoader worker (pid 12345) is killed by signal: Killed."
WORKER_BUS = "RuntimeError: DataLoader worker (pid(s) 12345, 6) is killed by signal: Bus error."


def _ids(diags):
    return {d.signature_id for d in diags}


def test_dataloader_worker_killed_is_oom_high_confidence():
    diags = diagnose(_ctx(stderr=WORKER_KILLED, exit_code=1))
    top = diags[0]
    assert top.signature_id == "torch_dataloader_worker_killed_oom"
    assert top.category == "oom"
    assert top.confidence == "high"
    assert any("num_workers" in f for f in top.suggested_fixes)


def test_worker_killed_corroborated_by_memory_ceiling():
    # host RAM at 95% strengthens the OOM story and lands in the evidence
    mem = _mem(used=15200.0, total=16000.0, rss=14000.0, count=8)
    diags = diagnose(_ctx(stderr=WORKER_KILLED, exit_code=1, memory=mem))
    oom = next(d for d in diags if d.category == "oom")
    assert oom.confidence == "high"
    assert oom.evidence.get("peak_mem_pct") is not None
    assert "corroborates" in oom.what_happened


def test_worker_killed_bus_error_routes_to_shm():
    diags = diagnose(_ctx(stderr=WORKER_BUS, exit_code=1))
    top = diags[0]
    assert top.category == "shm"
    assert "shm" in top.what_happened.lower() or "shared" in top.what_happened.lower()
    assert any("shm-size" in f for f in top.suggested_fixes)


def test_host_sigkill_without_torch_message_is_generic_oom():
    mem = _mem(used=15600.0, total=16000.0)
    diags = diagnose(_ctx(term_signal=9, memory=mem))
    oom = next(d for d in diags if d.category == "oom")
    assert oom.signature_id == "host_oom"
    assert oom.confidence == "high"  # SIGKILL + RAM at ceiling


def test_cuda_fork_signature():
    stderr = (
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. "
        "To use CUDA with multiprocessing, you must use the 'spawn' start method"
    )
    diags = diagnose(_ctx(stderr=stderr, exit_code=1))
    assert "fork_cuda" in {d.category for d in diags}
    fork = next(d for d in diags if d.category == "fork_cuda")
    assert any("spawn" in f for f in fork.suggested_fixes)


def test_unpicklable_spawn_signature():
    stderr = "AttributeError: Can't pickle local object 'build_loader.<locals>.<lambda>'"
    diags = diagnose(_ctx(stderr=stderr, exit_code=1))
    assert "pickle" in {d.category for d in diags}


def test_too_many_open_files_signature():
    stderr = "RuntimeError: received 0 items of ancdata"
    diags = diagnose(_ctx(stderr=stderr, exit_code=1))
    fds = next(d for d in diags if d.category == "fds")
    assert any("file_system" in f for f in fds.suggested_fixes)


def test_clean_run_yields_no_diagnosis():
    assert diagnose(_ctx(exit_code=0)) == []


def test_unknown_failure_falls_back_to_generic():
    diags = diagnose(_ctx(stderr="ValueError: something app-specific", exit_code=1))
    assert _ids(diags) == {"generic_nonzero_exit"}
    assert diags[0].confidence == "low"


def test_generic_catchall_suppressed_when_specific_signature_matches():
    # A failing run whose stderr matches a real signature should NOT also carry
    # the contradictory "no signature matched" catch-all.
    diags = diagnose(_ctx(stderr=WORKER_KILLED, exit_code=1))
    assert "generic_nonzero_exit" not in _ids(diags)


def test_dedup_keeps_one_per_category():
    # torch worker-killed (oom, high) and a bare SIGKILL (host_oom, oom) both
    # describe OOM -- only the most confident should survive in that category.
    mem = _mem(used=15600.0, total=16000.0)
    diags = diagnose(_ctx(stderr=WORKER_KILLED, term_signal=9, memory=mem))
    oom = [d for d in diags if d.category == "oom"]
    assert len(oom) == 1
    assert oom[0].signature_id == "torch_dataloader_worker_killed_oom"
