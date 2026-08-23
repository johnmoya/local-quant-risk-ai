"""Historical VaR: empirical quantile of the historical return series.

VaR_alpha = -Quantile(returns, 1 - alpha) * portfolio_value, reported as a
non-negative loss magnitude (see docs/math_reference.md).

Implemented in M2.
"""
