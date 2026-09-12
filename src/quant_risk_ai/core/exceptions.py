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
