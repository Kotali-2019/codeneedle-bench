"""Canonical scoring-policy representation used by runs and analysis."""
from __future__ import annotations

from dataclasses import dataclass

from .scorer import PASS_RATIO


@dataclass(frozen=True)
class ScoringPolicy:
    """Every switch that changes the meaning of a benchmark score."""

    relax_indent: bool = False
    count_comments: bool = True
    count_blank_lines: bool = False
    pass_ratio: float = PASS_RATIO

    def as_dict(self) -> dict[str, bool | float]:
        return {
            "relax_indent": self.relax_indent,
            "count_comments": self.count_comments,
            "count_blank_lines": self.count_blank_lines,
            "pass_ratio": self.pass_ratio,
        }

    @property
    def slug(self) -> str:
        indent = "content" if self.relax_indent else "strict"
        comments = "comments" if self.count_comments else "code-only"
        blanks = "blanks" if self.count_blank_lines else "no-blanks"
        threshold = f"pass-{self.pass_ratio * 100:g}pct".replace(".", "p")
        return f"{indent}-{comments}-{blanks}-{threshold}"

    @property
    def description(self) -> str:
        indent = "content-normalized indentation" if self.relax_indent else "strict indentation"
        comments = "comments counted" if self.count_comments else "code only"
        blanks = "blank lines counted" if self.count_blank_lines else "blank lines excluded"
        return f"{indent}; {comments}; {blanks}; {self.pass_ratio * 100:g}% pass threshold"


DEFAULT_SCORING_POLICY = ScoringPolicy()


def policy_from_dump(data: dict) -> ScoringPolicy:
    """Parse and validate the scoring policy recorded in a result dump."""
    raw = data.get("scoring")
    if not isinstance(raw, dict):
        raise ValueError("missing `scoring` policy metadata")

    required = {
        "relax_indent": bool,
        "count_comments": bool,
        "count_blank_lines": bool,
    }
    for name, expected_type in required.items():
        if name not in raw:
            raise ValueError(f"scoring policy is missing `{name}`")
        if not isinstance(raw[name], expected_type):
            raise ValueError(f"scoring policy `{name}` must be a boolean")

    ratio = raw.get("pass_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise ValueError("scoring policy `pass_ratio` must be a number")
    if not 0 < float(ratio) <= 1:
        raise ValueError("scoring policy `pass_ratio` must be in (0, 1]")

    return ScoringPolicy(
        relax_indent=raw["relax_indent"],
        count_comments=raw["count_comments"],
        count_blank_lines=raw["count_blank_lines"],
        pass_ratio=float(ratio),
    )
