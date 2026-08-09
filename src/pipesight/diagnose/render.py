"""Text rendering for `pipesight diagnose`."""

from __future__ import annotations

from pipesight.diagnose.diagnose import Diagnosis

_CONFIDENCE_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


def render_diagnosis(d: Diagnosis, index: int) -> str:
    lines = [
        f"  [{index}] {d.title}  (confidence: {_CONFIDENCE_LABEL[d.confidence]})",
        f"      What happened: {d.what_happened}",
    ]
    if d.likely_causes:
        lines.append("      Likely causes:")
        lines += [f"        - {c}" for c in d.likely_causes]
    if d.suggested_fixes:
        lines.append("      Suggested fixes:")
        lines += [f"        - {f}" for f in d.suggested_fixes]
    return "\n".join(lines)


def render_diagnoses(diagnoses: list[Diagnosis], *, failed: bool) -> str:
    if not diagnoses:
        if not failed:
            return (
                "== Diagnosis ==\n"
                "  The run completed successfully (no failure detected) -- nothing to diagnose."
            )
        return (
            "== Diagnosis ==\n"
            "  The run failed, but no signature matched and no failure signals were captured.\n"
            "  Re-run under `pipesight diagnose -- <command>` (which tees stderr), or pass the\n"
            "  training log with `--log <file>`, so there's output to match against."
        )
    blocks = [render_diagnosis(d, i) for i, d in enumerate(diagnoses, 1)]
    header = (
        "== Diagnosis ==\n"
        "  (advisory: signatures are strong hints from known failure patterns, not proof)\n"
    )
    return header + "\n\n".join(blocks)
