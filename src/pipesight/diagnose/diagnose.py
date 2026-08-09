"""Crash/failure triage for a pipeline run.

Deliberately *separate* from `analysis.recommend`: recommendations are about a
healthy run wasting GPU time and carry an estimated-savings figure. A crash has
no "savings" -- the run died, and the answer is "here's why and what to try."
So diagnoses are their own result type, produced by pattern-matching a run's
failure signals (terminating signal, exit code, a tail of stderr) against a
playbook of known signatures, and corroborated with the memory timeline when
one is available.

Everything here is advisory: a signature matching is strong evidence, not
proof. Output says "usually means", never "definitely is".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pipesight.analysis.memory import MemoryReport, memory_from_samples
from pipesight.trace.schema import Trace

Confidence = Literal["high", "medium", "low"]
_CONFIDENCE_RANK: dict[Confidence, int] = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Diagnosis:
    signature_id: str
    # Coarse failure family; used to de-duplicate when several signatures
    # describe the same underlying event (keep the most confident per family).
    category: str
    title: str
    confidence: Confidence
    what_happened: str
    likely_causes: list[str]
    suggested_fixes: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnoseContext:
    """Everything the signatures get to look at."""

    stderr: str
    exit_code: int | None
    term_signal: int | None
    memory: MemoryReport | None

    @property
    def killed_by_signal(self) -> bool:
        return self.term_signal is not None

    @property
    def failed(self) -> bool:
        return self.term_signal is not None or bool(self.exit_code)


# A signature inspects the context and returns a Diagnosis if it matches.
Signature = Callable[[DiagnoseContext], Diagnosis | None]


def diagnose(ctx: DiagnoseContext, signatures: list[Signature] | None = None) -> list[Diagnosis]:
    """Run every signature over the context, keep the most confident match per
    failure category, and return them most-confident-first."""
    if signatures is None:
        from pipesight.diagnose.signatures import default_signatures

        signatures = default_signatures()

    best_by_category: dict[str, Diagnosis] = {}
    for sig in signatures:
        d = sig(ctx)
        if d is None:
            continue
        existing = best_by_category.get(d.category)
        more_confident = existing is None or (
            _CONFIDENCE_RANK[d.confidence] > _CONFIDENCE_RANK[existing.confidence]
        )
        if more_confident:
            best_by_category[d.category] = d

    # The "unknown" catch-all only earns its place when nothing specific
    # matched -- otherwise it contradicts the real diagnosis ("no signature
    # matched" printed right after one did).
    if len(best_by_category) > 1:
        best_by_category.pop("unknown", None)

    return sorted(
        best_by_category.values(),
        key=lambda d: _CONFIDENCE_RANK[d.confidence],
        reverse=True,
    )


def context_from_trace(trace: Trace) -> DiagnoseContext:
    return DiagnoseContext(
        stderr=trace.meta.stderr_tail or "",
        exit_code=trace.meta.exit_code,
        term_signal=trace.meta.term_signal,
        memory=memory_from_samples(trace.samples),
    )
