"""Ranked, actionable recommendations built from a Trace's spans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pipesight.analysis.idle import gpu_idle_from_spans
from pipesight.analysis.memory import MemoryReport, memory_from_samples
from pipesight.analysis.overlap import detect_cross_iteration_overlap
from pipesight.trace.schema import Span, Trace

RecommendationKind = Literal["increase_workers", "pipeline_overlap", "gpu_underutilized"]


@dataclass
class Recommendation:
    kind: RecommendationKind
    title: str
    detail: str
    estimated_savings_ns: int
    evidence: dict[str, Any] = field(default_factory=dict)


def recommend_worker_count(
    spans: list[Span], physical_cores: int, memory: MemoryReport | None = None
) -> Recommendation | None:
    if not spans or physical_cores <= 1:
        return None
    cpu_total = sum(s.duration_ns for s in spans if s.device == "cpu")
    gpu_total = sum(s.duration_ns for s in spans if s.device == "gpu")
    idle = gpu_idle_from_spans(spans)
    if idle.idle_pct <= 40.0 or cpu_total <= gpu_total:
        return None

    evidence: dict[str, Any] = {
        "cpu_total_ns": cpu_total,
        "gpu_total_ns": gpu_total,
        "idle_pct": idle.idle_pct,
    }
    base_detail = (
        f"CPU-bound stages account for {cpu_total / 1e9:.1f}s of work vs "
        f"{gpu_total / 1e9:.1f}s of GPU work, and the GPU is idle "
        f"{idle.idle_pct:.0f}% of the traced window. Running multiple worker "
        f"processes (e.g. a ProcessPoolExecutor) lets one worker's CPU-bound "
        f"phase overlap another's GPU-bound phase on the same GPU. Bound worker "
        f"count by *physical* cores ({physical_cores}), not logical -- "
        f"hyperthreads don't add real CPU-bound throughput."
    )

    # Memory-aware branch: "add workers" is actively harmful advice if there's
    # no RAM headroom for them -- that's exactly how a dataloader OOMs and you
    # get "DataLoader worker (pid N) killed by signal: Killed". Only override
    # when we actually measured host memory nearing the ceiling.
    if memory is not None and memory.near_ceiling and memory.peak_used_pct is not None:
        evidence["peak_mem_pct"] = memory.peak_used_pct
        evidence["headroom_mb"] = memory.headroom_mb
        fit_note = ""
        per_proc = memory.rss_per_proc_mb
        if per_proc and memory.headroom_mb is not None:
            extra = int(memory.headroom_mb // per_proc)
            evidence["rss_per_proc_mb"] = per_proc
            fit_note = (
                f" At the observed ~{per_proc:.0f} MB per process, only ~{extra} more "
                f"would fit before the OOM killer steps in."
            )
        return Recommendation(
            kind="increase_workers",
            title="CPU-bound, but host RAM is near the ceiling -- scale workers cautiously",
            detail=(
                f"{base_detail}\n\n      CAUTION: peak host RAM hit "
                f"{memory.peak_used_pct:.0f}% of {(memory.sys_total_mb or 0) / 1024:.1f} GiB "
                f"during this run. Adding workers naively here is how you get a "
                f"'DataLoader worker killed' OOM.{fit_note} Raise the worker/num_workers "
                f"count in small steps while watching RSS, and cut per-worker memory "
                f"(smaller prefetch/buffers, lazy loading, drop persistent_workers) before "
                f"scaling out."
            ),
            estimated_savings_ns=idle.idle_ns,
            evidence=evidence,
        )

    detail = base_detail
    if memory is not None and memory.has_data and memory.peak_used_pct is not None:
        evidence["peak_mem_pct"] = memory.peak_used_pct
        per_proc = memory.rss_per_proc_mb
        if per_proc and memory.headroom_mb is not None:
            extra = int(memory.headroom_mb // per_proc)
            detail += (
                f" Memory headroom looks comfortable (peak {memory.peak_used_pct:.0f}% of "
                f"{(memory.sys_total_mb or 0) / 1024:.1f} GiB); roughly {extra} more workers "
                f"of the current ~{per_proc:.0f} MB size would fit."
            )
        else:
            detail += (
                f" Peak host RAM was {memory.peak_used_pct:.0f}% of total -- headroom "
                f"looks fine for more workers."
            )

    return Recommendation(
        kind="increase_workers",
        title=f"Increase parallel workers (up to {physical_cores} physical cores)",
        detail=detail,
        estimated_savings_ns=idle.idle_ns,
        evidence=evidence,
    )


def recommend_overlap(spans: list[Span]) -> list[Recommendation]:
    recs = []
    for opp in detect_cross_iteration_overlap(spans):
        if opp.note:  # informational only (e.g. gpu+gpu pair) -- not a real savings claim
            continue
        snippet = (
            f'StageSpec("{opp.tail_stage}", {opp.tail_stage}_fn, device="{opp.tail_device}"),\n'
            f'        StageSpec("{opp.head_stage}", {opp.head_stage}_fn, '
            f'device="{opp.head_device}"),'
        )
        recs.append(
            Recommendation(
                kind="pipeline_overlap",
                title=f'Overlap "{opp.tail_stage}" with the next item\'s "{opp.head_stage}"',
                detail=(
                    f'"{opp.tail_stage}" finishes strictly before the next item\'s '
                    f'"{opp.head_stage}" starts in {opp.evidence_fraction:.0%} of '
                    f"{opp.pairs_examined} consecutive item pairs -- these can run "
                    f"concurrently. Wire both stages into a pipesight Pipeline:\n\n"
                    f"        from pipesight.pipeline import Pipeline, StageSpec\n"
                    f"        stages = [\n        {snippet}\n        ]\n"
                    f"        with Pipeline(stages, profiler=prof) as p:\n"
                    f"            for result in p.run(items): ..."
                ),
                estimated_savings_ns=opp.estimated_savings_ns,
                evidence={
                    "tail_stage": opp.tail_stage,
                    "head_stage": opp.head_stage,
                    "evidence_fraction": opp.evidence_fraction,
                },
            )
        )
    return recs


def rank(recs: list[Recommendation]) -> list[Recommendation]:
    return sorted(recs, key=lambda r: r.estimated_savings_ns, reverse=True)


def build_recommendations(
    trace: Trace, physical_cores: int | None = None
) -> list[Recommendation]:
    spans = trace.spans
    recs: list[Recommendation] = []

    cores = physical_cores or trace.meta.cpu_count_physical or 1
    memory = memory_from_samples(trace.samples)
    worker_rec = recommend_worker_count(spans, cores, memory=memory)
    if worker_rec:
        recs.append(worker_rec)

    recs.extend(recommend_overlap(spans))

    if not recs and spans:
        idle = gpu_idle_from_spans(spans)
        if idle.idle_pct > 40.0:
            recs.append(
                Recommendation(
                    kind="gpu_underutilized",
                    title="GPU is idle a significant fraction of the time",
                    detail=(
                        f"GPU idle {idle.idle_pct:.0f}% of the traced window, but no "
                        f"specific worker-count or overlap pattern was detected. Check "
                        f"whether stages have item_id set consistently for deeper analysis."
                    ),
                    estimated_savings_ns=idle.idle_ns,
                )
            )

    return rank(recs)
