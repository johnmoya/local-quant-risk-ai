"""Mandatory post-hoc guard against claims a RiskResult cannot support.

The numeric check (numeric_check.py) makes sure every number in an
explanation is accounted for; it cannot see a sentence with no number in
it. A RiskResult carries one figure for the whole position, so it says
nothing about which asset contributes the risk, how the assets co-move,
how much diversification is worth, whether the model is well calibrated,
or what anyone should do. An explanation that asserts any of that is
narrating something that was never computed.

The guard is lexical and deterministic: a fixed list of stems per
category, matched case-insensitively at word boundaries, and one match
rejects the explanation (502, category named) exactly as a stray number
does. A negation is rejected too ("this does not reflect diversification"):
it is safer to refuse a correct sentence than to parse one. What it
cannot catch is a paraphrase that avoids every stem ("spreading the
holdings lowers the loss"); that gap is stated in docs/architecture.md.
Using an LLM as the judge instead was ruled out: it is not deterministic,
and it would put a model back in the verification path.

Applied to every explanation, single-asset or portfolio.
"""

from __future__ import annotations

import re

from quant_risk_ai.core.exceptions import UnsupportedClaimError

_STEMS: dict[str, tuple[str, ...]] = {
    "attribution": (r"contribut\w*", r"marginal\w*", r"component\w*"),
    "diversification": (r"diversif\w*",),
    "correlation": (r"correlat\w*", r"hedg\w*"),
    # Not "accura-" or "reliab-" (removed before v1.1.0): measured against
    # qwen3:8b they never caught the model vouching for itself, only sound
    # caveats drawn from a flagged diagnostic ("two dates were dropped, which
    # may affect the accuracy of the result"), rejecting about a fifth of the
    # explanations of historical results with a sparse tail.
    "model_quality": (r"calibrat\w*", r"backtest\w*"),
    "advice": (r"recommend\w*", r"should\s+(?:buy|sell|reduce|increase)"),
    "guarantee": (r"guarantee\w*",),
}

# "uncorrelated", "uncalibrated", "non-diversified" make the same claim as
# the bare stem, so the common negating prefixes are part of the word.
_PATTERNS: dict[str, re.Pattern[str]] = {
    category: re.compile(rf"\b(?:un|in|non-?)?(?:{'|'.join(stems)})\b", re.IGNORECASE)
    for category, stems in _STEMS.items()
}


def verify_no_unsupported_claims(text: str) -> None:
    """Raise UnsupportedClaimError naming the first category whose stems
    appear in `text`, and the word that matched.

    Like verify_numeric_consistency, this is the gate itself, meant to be
    called unconditionally on the return path of every explanation.
    """
    for category, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            raise UnsupportedClaimError(category=category, term=match.group())
