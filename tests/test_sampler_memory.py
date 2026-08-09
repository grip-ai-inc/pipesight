from __future__ import annotations

import os
import time

from pipesight.profiling.cpu import sample_process_tree_rss_mb, sample_system_memory_mb
from pipesight.profiling.sampler import SamplerThread


def test_system_memory_helper():
    used, total = sample_system_memory_mb()
    assert total > 0
    assert 0 < used <= total


def test_process_tree_rss_current_process():
    tree = sample_process_tree_rss_mb(os.getpid())
    assert tree is not None
    rss_mb, count = tree
    assert rss_mb > 0
    assert count >= 1


def test_process_tree_rss_missing_pid_returns_none():
    # PID 2**31-1 is effectively guaranteed not to exist
    assert sample_process_tree_rss_mb(2**31 - 1) is None


def test_sampler_records_memory_by_default():
    sampler = SamplerThread(interval_s=0.02, use_gpu=False)
    sampler.root_pid = os.getpid()
    sampler.start()
    time.sleep(0.12)
    samples = sampler.stop()

    assert samples
    mem_samples = [s for s in samples if s.sys_mem_used_mb is not None]
    assert mem_samples, "expected system memory to be sampled"
    with_proc = [s for s in samples if s.proc_rss_mb is not None]
    assert with_proc, "expected process-tree RSS once root_pid was set"
    assert with_proc[0].proc_count >= 1


def test_sampler_memory_can_be_disabled():
    sampler = SamplerThread(interval_s=0.02, use_gpu=False, sample_memory=False)
    sampler.start()
    time.sleep(0.08)
    samples = sampler.stop()
    assert samples
    assert all(s.sys_mem_used_mb is None for s in samples)
