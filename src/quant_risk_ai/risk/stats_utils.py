"""Shared statistical primitives used across the VaR/ES modules
(annualization, volatility/covariance estimation, quantile helpers), so the
method-specific modules don't duplicate math.

Pluggable simulation-distribution hook (design note for M4 / future work):
var_monte_carlo.py's default sampler is normal-via-Cholesky. A historical
bootstrap sampler (drawing with replacement from realized returns instead of
an assumed distribution) is a natural extension and should be added here as
a second `sample_*` function with the same signature, so var_monte_carlo.py
can select between them by a `distribution` parameter without changing its
own structure. Not implemented in v1 — noted so it's a drop-in addition
later, not a redesign.

Implemented starting M2.
"""
