"""Mandatory post-hoc numeric-consistency check for generated explanations.

This is the concrete, tested safeguard for the project's core principle
(the LLM never computes risk figures, only narrates them): it extracts every
numeric token from the LLM's generated text and verifies each one against
the source RiskResult's fields (value, confidence_level, horizon_days,
portfolio_value, etc.), within a defined tolerance for formatting/rounding.

This is a required gate in explain.py's return path, not an optional or
best-effort diagnostic: an explanation whose numbers don't reconcile with
the source RiskResult must not be returned to the caller. M7's test suite
must cover both the pass and reject paths for this check.

Implemented in M7.
"""
