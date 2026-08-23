# Mathematical Reference

This document is completed incrementally as each method is implemented.
Sections marked TODO are filled in during the milestone noted.

## Sign convention (binding for the whole project)

**VaR and Expected Shortfall are always reported as a non-negative number
representing the magnitude of a potential loss**, regardless of method
(Historical, Parametric, Monte Carlo) or metric (VaR, ES).

- A VaR of `1,234.56` means "a loss of up to 1,234.56 is expected not to be
  exceeded at the stated confidence level" — never a negative number.
- This holds even when the underlying return distribution has positive
  expected value, in which case the raw quantile/tail-mean of the return
  distribution would itself be negative before the sign flip.
- This convention is enforced in code, not just documented: constructing a
  `RiskResult` (`src/quant_risk_ai/risk/results.py`) with a negative `value`
  raises `ValueError`. See `tests/unit/risk/test_results.py::test_negative_value_is_rejected`
  and `test_expected_shortfall_metric_also_enforces_sign` for the tests
  pinning this down.

Rationale: mixed sign conventions (some codebases report VaR as a negative
return, others as a positive loss) are a well-known source of silent bugs
when composing VaR figures with P&L or capital figures downstream. Fixing
one convention project-wide, and enforcing it at the data-contract level
rather than trusting every call site, removes that entire class of bug.

## Return convention

TODO (M1): log returns vs. simple returns — decision and rationale, plus the
explicit missing-data handling policy.

## Historical VaR

TODO (M2): empirical quantile definition, worked example, minimum sample
size guidance.

## Historical Expected Shortfall

TODO (M2): empirical tail-mean definition, worked example.

## Parametric (Variance-Covariance) VaR

TODO (M3): normal-distribution assumption, closed-form derivation, and its
known limitation under fat tails / skew.

## Parametric Expected Shortfall

TODO (M3): closed-form normal ES derivation.

## Monte Carlo VaR

TODO (M4): simulation methodology (normal via Cholesky, default), simulation
count vs. accuracy/runtime tradeoff, RNG seeding for reproducibility.

## Monte Carlo Expected Shortfall

TODO (M4): tail-mean over simulated P&L.

## Time horizon scaling

TODO: the √t scaling assumption from 1-day to t-day VaR, and why it's an
approximation (i.i.d., no autocorrelation) rather than exact.

## Backtesting

TODO (M5): Kupiec POF test, Christoffersen independence and conditional
coverage tests, Basel traffic-light zones, violation ratio — definitions and
worked examples.
