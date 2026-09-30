"""Tests for llm/facts.py: the prompt and the numeric check's pool are the
same facts, in both directions (every number the prompt shows is in the
pool, and every pooled fact is shown in the prompt)."""

from datetime import date
from typing import Any

import numpy as np
import pytest

from quant_risk_ai.llm.facts import Unit, build_fact_sheet
from quant_risk_ai.llm.numeric_check import (
    _extract_dates,
    _extract_numbers,
    expected_numbers,
    verify_numeric_consistency,
)
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
from quant_risk_ai.risk.expected_shortfall import (
    historical_expected_shortfall,
    monte_carlo_expected_shortfall,
    parametric_expected_shortfall,
)
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from quant_risk_ai.risk.var_historical import historical_var
from quant_risk_ai.risk.var_monte_carlo import monte_carlo_var
from quant_risk_ai.risk.var_parametric import parametric_var
from tests.unit.risk._helpers import make_asset_returns

_SERIES = make_asset_returns(
    list(np.random.default_rng(7).normal(0.0005, 0.02, 300)), asset_id="AAPL"
)
_COMMON: dict[str, Any] = {
    "alpha": 0.99,
    "position_value": 250_000.0,
    "horizon_days": 10,
    "as_of": date(2026, 3, 5),
}


def _single_asset_results() -> list[RiskResult]:
    return [
        historical_var(_SERIES, **_COMMON),
        parametric_var(_SERIES, **_COMMON),
        monte_carlo_var(_SERIES, seed=3, n_simulations=5_000, **_COMMON),
        historical_expected_shortfall(_SERIES, **_COMMON),
        parametric_expected_shortfall(_SERIES, **_COMMON),
        monte_carlo_expected_shortfall(_SERIES, seed=3, n_simulations=5_000, **_COMMON),
    ]


def _facts_block(prompt: str) -> str:
    """The rendered facts, without the instructions around them."""
    return prompt.split("\n\n")[-2]


def _prompt_figures(result: RiskResult) -> tuple[list[date | str], list[float]]:
    dates, remaining = _extract_dates(_facts_block(build_explanation_prompt(result)))
    return dates, _extract_numbers(remaining)


@pytest.fixture(params=_single_asset_results(), ids=lambda r: f"{r.method.value}-{r.metric.value}")
def result(request: pytest.FixtureRequest) -> RiskResult:
    return request.param


def test_every_number_in_the_prompt_is_a_fact(result):
    facts = build_fact_sheet(result)
    forms = {form for fact in facts.numbers for form in fact.forms}

    _, numbers = _prompt_figures(result)

    assert numbers, "the prompt shows no numbers at all"
    assert [n for n in numbers if n not in forms] == []


def test_every_fact_is_shown_in_the_prompt(result):
    facts = build_fact_sheet(result)

    dates, numbers = _prompt_figures(result)

    assert [f for f in facts.numbers if f.forms[0] not in numbers] == []
    assert set(dates) == set(facts.dates)


def test_the_prompt_read_back_passes_the_numeric_check(result):
    verify_numeric_consistency(_facts_block(build_explanation_prompt(result)), result)


def test_the_confidence_level_is_a_rate_with_its_complement():
    facts = build_fact_sheet(_single_asset_results()[0])

    rates = [fact for fact in facts.numbers if fact.unit is Unit.RATE]

    assert rates[0].forms == (0.99, pytest.approx(0.01))


def test_nested_metadata_is_neither_shown_nor_pooled():
    result = RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.VAR,
        value=1234.56,
        confidence_level=0.99,
        horizon_days=1,
        portfolio_value=100_000.0,
        as_of=date(2026, 8, 21),
        n_observations=250,
        asset_ids=["AAPL"],
        metadata={"tail_size": 3, "weights": [0.37], "by_asset": {"AAPL": {"n": 777}}},
    )

    prompt = build_explanation_prompt(result)
    pool = expected_numbers(build_fact_sheet(result))

    assert "tail_size: 3" in prompt
    assert "0.37" not in prompt and "777" not in prompt
    assert 0.37 not in pool and 37.0 not in pool and 777.0 not in pool
