"""Typed exceptions shared across layers, so the API layer can map them to
the right HTTP status instead of leaking raw pandas/numpy/httpx errors.

Populated incrementally as each layer's edge cases are implemented
(started M1 for the data layer; risk/api/llm exceptions follow in later
milestones).
"""

from __future__ import annotations


class QuantRiskAIError(Exception):
    """Base class for all project-specific exceptions."""


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
