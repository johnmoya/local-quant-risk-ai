"""Orchestrates RiskResult -> prompt -> Ollama call -> validated explanation.

Every explanation returned by this module MUST have passed
numeric_check.py's verification against its source RiskResult first
(mandatory, per docs/roadmap.md M7 — not an optional/best-effort step).
On Ollama failure or a failed numeric check, this module must fail
explicitly rather than silently returning unverified text.

Implemented in M7.
"""
