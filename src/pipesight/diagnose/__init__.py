"""Crash/failure triage for pipeline runs: match a run's failure signals
(terminating signal, exit code, a tail of stderr) against a playbook of known
signatures, corroborated with the memory timeline. Separate from the
perf-oriented `analysis.recommend` engine on purpose -- see `diagnose.py`.
"""

from pipesight.diagnose.diagnose import (
    DiagnoseContext,
    Diagnosis,
    context_from_trace,
    diagnose,
)

__all__ = ["Diagnosis", "DiagnoseContext", "diagnose", "context_from_trace"]
