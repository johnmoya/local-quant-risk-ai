"""Expected Shortfall (CVaR) for the historical, parametric, and Monte Carlo
methods.

- Empirical (historical / Monte Carlo): mean of losses beyond the VaR cutoff.
- Closed-form normal: ES_alpha = -(mu - sigma * phi(z_(1-alpha)) / (1-alpha)).

Always non-negative, same convention as VaR (see docs/math_reference.md).

Implemented in M2 (empirical) and M3 (closed-form normal).
"""
