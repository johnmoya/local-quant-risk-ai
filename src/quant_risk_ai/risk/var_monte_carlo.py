"""Monte Carlo VaR.

Default distribution: normal, sampled via Cholesky decomposition of the
covariance matrix (single-asset in v1, so this reduces to sampling from
N(mu, sigma^2) directly). The empirical quantile of the simulated P&L is
then computed the same way as in var_historical.py.

Reproducibility: simulations must take an explicit RNG seed.

Implemented in M4.
"""
