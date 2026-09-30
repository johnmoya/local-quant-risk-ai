"""Typed exceptions shared across layers, so the API layer can map them to
the right HTTP status instead of leaking raw pandas/numpy/httpx errors.

Populated incrementally as each layer's edge cases are implemented
(started M1 for the data layer; risk/api/llm exceptions follow in later
milestones).
"""

from __future__ import annotations


class QuantRiskAIError(Exception):
    """Base class for all project-specific exceptions."""


class InvalidParameterError(QuantRiskAIError):
    """A caller-supplied scalar parameter, or the resulting RiskResult
    itself, fails a validation rule — e.g. alpha outside the open interval
    (0, 1), a non-positive horizon_days, or a RiskResult invariant like a
    negative value or empty asset_ids.

    Distinct from DataValidationError (which is about the shape of a data
    *series* — dates, columns) and from InsufficientDataError (which is
    about *how much* data there is, not the validity of a single
    parameter or an already-computed result).
    """


class DataValidationError(QuantRiskAIError):
    """Input data fails a structural validation check: duplicate dates,
    unparseable dates, missing required columns, or non-positive prices
    where a log transform requires them.
    """


class InsufficientDataError(QuantRiskAIError):
    """Not enough data to perform the requested calculation — e.g. fewer
    than two chronologically adjacent, non-missing price observations to
    compute a single return.
    """


class InsufficientSampleSizeError(InsufficientDataError):
    """There IS data, but not enough of it for the requested confidence
    level's quantile to be backed by even one real tail observation — a
    more specific condition than InsufficientDataError's base case of zero
    valid data points. See risk/stats_utils.py:min_required_observations.
    """


class SingularCovarianceError(QuantRiskAIError):
    """A portfolio's covariance matrix cannot be factorised: it is singular
    or too ill-conditioned for a Cholesky decomposition.

    Distinct from InsufficientSampleSizeError, which is about *how much*
    data backs an estimate: a covariance matrix can be singular with any
    amount of data, because two perfectly collinear assets or one with zero
    variance do it on their own. Raised where the failure actually occurs —
    the factorisation — and carries the condition number, so a caller can
    tell rank deficiency by construction from mere ill-conditioning. See
    risk/covariance.py.
    """


class PortfolioMethodsFailedError(QuantRiskAIError):
    """One or more of the (method, metric) pairs requested together failed.

    All or nothing: no result is returned when any pair fails, so a report
    can never go out silently missing a method. Every pair is still
    computed, so the error lists *all* failures, not just the first, and
    names the pairs that succeeded but were withheld — a caller whose Monte
    Carlo hit a singular covariance learns that dropping it returns the
    rest.
    """

    def __init__(self, failures: list[dict[str, str]], withheld: list[dict[str, str]]):
        self.failures = failures
        self.withheld = withheld
        names = ", ".join(f"{f['method']}/{f['metric']}" for f in failures)
        super().__init__(f"{len(failures)} requested calculation(s) failed: {names}")


class LLMError(QuantRiskAIError):
    """Base class for LLM-layer (quant_risk_ai.llm) failures. Split into
    subclasses below rather than used directly, so api/main.py can map each
    failure mode to its own HTTP status instead of a one-size-fits-all 422
    (an unreachable Ollama instance is not a client input error, and
    neither is a generated explanation failing its numeric check).
    """


class LLMUnavailableError(LLMError):
    """The Ollama backend could not be reached, timed out, returned a
    non-2xx response, or returned a response this client doesn't know how
    to parse. Always the local Ollama service's fault, never the
    caller's — see api/main.py's 503 mapping.
    """


class NumericConsistencyError(LLMError):
    """A generated explanation contains a number that could not be
    reconciled with its source RiskResult (see
    quant_risk_ai.llm.numeric_check) — the model altered, recomputed, or
    invented a figure. This is the mandatory post-hoc safeguard from
    docs/roadmap.md M7: an explanation that fails this check must never
    reach the caller.
    """


class UnsupportedClaimError(LLMError):
    """A generated explanation asserts something its source RiskResult
    cannot support — attribution of risk to an asset, correlation,
    diversification, model quality, advice, a guarantee (see
    quant_risk_ai.llm.claims). Same status as NumericConsistencyError,
    502, with the category named so a caller can tell the two apart.
    """

    def __init__(self, category: str, term: str):
        self.category = category
        self.term = term
        super().__init__(
            f"Generated explanation makes a claim the RiskResult does not support "
            f"({category}: {term!r})"
        )
