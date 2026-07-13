"""Detects stages that are strictly sequential across loop iterations
(correlated by `item_id`) but could overlap -- the generalization of
"segment N+1's decode/SLAM could start while segment N's hand-pose/upload is
still running", the finding this whole tool is built to generalize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipesight.trace.schema import Span


@dataclass
class OverlapOpportunity:
    tail_stage: str
    head_stage: str
    tail_device: str
    head_device: str
    evidence_fraction: float
    pairs_examined: int
    estimated_savings_ns: int
    note: str = ""


def _item_span_groups(spans: list[Span]) -> list[tuple[Any, list[Span]]]:
    """Groups spans by item_id, preserving first-seen order. Spans without
    an item_id are excluded -- overlap detection needs loop correlation."""
    order: list[Any] = []
    groups: dict[Any, list[Span]] = {}
    for s in spans:
        if s.item_id is None:
            continue
        if s.item_id not in groups:
            groups[s.item_id] = []
            order.append(s.item_id)
        groups[s.item_id].append(s)
    return [(item_id, groups[item_id]) for item_id in order]


def detect_cross_iteration_overlap(
    spans: list[Span], min_fraction: float = 0.8
) -> list[OverlapOpportunity]:
    items = _item_span_groups(spans)
    if len(items) < 2:
        return []

    pair_count = 0
    sequential_count = 0
    tail_names: dict[str, int] = {}
    head_names: dict[str, int] = {}
    tail_devices: dict[str, str] = {}
    head_devices: dict[str, str] = {}
    savings_ns = 0

    for (_, cur_spans), (_, next_spans) in zip(items, items[1:]):
        pair_count += 1
        tail_span = max(cur_spans, key=lambda s: s.end_ns)
        head_span = min(next_spans, key=lambda s: s.start_ns)
        if head_span.start_ns >= tail_span.end_ns:
            sequential_count += 1
            tail_names[tail_span.name] = tail_names.get(tail_span.name, 0) + 1
            head_names[head_span.name] = head_names.get(head_span.name, 0) + 1
            tail_devices[tail_span.name] = tail_span.device
            head_devices[head_span.name] = head_span.device
            savings_ns += min(tail_span.duration_ns, head_span.duration_ns)

    if pair_count == 0 or not tail_names:
        return []
    fraction = sequential_count / pair_count
    if fraction < min_fraction:
        return []

    modal_tail = max(tail_names, key=lambda n: tail_names[n])
    modal_head = max(head_names, key=lambda n: head_names[n])
    tail_device = tail_devices[modal_tail]
    head_device = head_devices[modal_head]

    note = ""
    if tail_device == "gpu" and head_device == "gpu":
        note = (
            "Both stages are GPU-bound on what's presumably a single physical GPU -- "
            "overlapping them won't help under the default concurrency=1; look at "
            "reducing per-stage GPU compute instead."
        )

    return [
        OverlapOpportunity(
            tail_stage=modal_tail,
            head_stage=modal_head,
            tail_device=tail_device,
            head_device=head_device,
            evidence_fraction=fraction,
            pairs_examined=pair_count,
            estimated_savings_ns=savings_ns,
            note=note,
        )
    ]
