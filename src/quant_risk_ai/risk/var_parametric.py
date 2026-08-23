"""Parametric (variance-covariance / delta-normal) VaR.

Assumes returns ~ N(mu, sigma^2): VaR_alpha = -(mu + z_(1-alpha) * sigma) * V.
v1 is single-asset, so sigma is the sample return volatility directly (no
covariance matrix yet — that arrives with multi-asset support in M11).

Implemented in M3.
"""
