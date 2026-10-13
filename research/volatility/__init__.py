"""M13/M14 volatility research: data, fitting and orchestration.

The pure math (volatility recursions, conditional VaR/ES, evaluation
statistics) lives in `quant_risk_ai.risk`; this package loads data, fits
parameters with external libraries and runs the walk-forward. Design and
pre-registration: `docs/design_m13.md`.
"""
