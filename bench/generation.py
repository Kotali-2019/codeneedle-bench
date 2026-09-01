"""Benchmark-generation identifiers shared by producers and consumers.

A result is scientifically comparable only when the model saw the same prompt
contract and its stored aggregates came from the same scorer semantics.  Dump
schema versions are not sufficient for that: schema v2 existed across prompt
and scoring changes, which allowed incompatible historical runs into the same
leaderboard.

Increment these identifiers whenever the corresponding semantics change.  The
values are deliberately descriptive so stale-result diagnostics are useful to
people reading a JSON dump by hand.
"""
from __future__ import annotations


PROMPT_VERSION = "signature-anchor-v1"
SCORER_VERSION = "blankless-line-lcs-v2"
GENERATION_FIELD = "benchmark_generation"


def current_generation() -> dict[str, str]:
    """Return a fresh JSON-serializable description of the current generation."""
    return {
        "prompt": PROMPT_VERSION,
        "scorer": SCORER_VERSION,
    }


def generation_problem(data: dict) -> str | None:
    """Explain why a dump cannot be compared with the current generation.

    Unknown is intentionally incompatible.  Guessing from ``schema_version``
    would recreate the original bug because schema v2 spans more than one
    prompt/scorer generation.
    """
    raw = data.get(GENERATION_FIELD)
    if not isinstance(raw, dict):
        return f"missing `{GENERATION_FIELD}` metadata"

    expected = current_generation()
    missing = [name for name in expected if not raw.get(name)]
    if missing:
        return "missing generation value(s): " + ", ".join(missing)

    mismatches = [
        f"{name}={raw[name]!r} (current {value!r})"
        for name, value in expected.items()
        if raw.get(name) != value
    ]
    if mismatches:
        return "outdated generation: " + "; ".join(mismatches)
    return None
