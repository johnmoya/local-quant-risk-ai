"""The portfolio path of the explanation layer (M11.6): the fact sheet and
prompt for results with more than one asset, the edge validation of what
the caller resubmits, and the fabricated explanations of the M11 design
(docs/design_m11.md §7), each run through the real checks."""

import math
import re
from datetime import date

import httpx
import pytest

from quant_risk_ai import config
from quant_risk_ai.core.exceptions import (
    InvalidParameterError,
    NumericConsistencyError,
    UnsupportedClaimError,
)
from quant_risk_ai.llm.explain import generate_explanation, prompt_token_bound
from quant_risk_ai.llm.facts import (
    LISTED_DROPPED_DATES,
    TOP_POSITIONS,
    FactSheet,
    Unit,
    build_fact_sheet,
)
from quant_risk_ai.llm.numeric_check import (
    _extract_dates,
    _extract_tokens,
    _matches,
    _remove_names,
    verify_numeric_consistency,
)
from quant_risk_ai.llm.ollama_client import OllamaClient
from quant_risk_ai.llm.prompt_templates import build_explanation_prompt
from quant_risk_ai.risk.results import RiskMethod, RiskMetric, RiskResult
from tests.unit.llm._portfolio import ENGINES, make_portfolio, run

_TIGHT = {"rel_tol": 1e-12, "abs_tol": 1e-12}


def _facts_block(prompt: str) -> str:
    return prompt.split("\n\n")[-2]


def _prompt_tokens(result: RiskResult):
    facts = build_fact_sheet(result)
    block = _remove_names(_facts_block(build_explanation_prompt(result)), facts.names)
    dates, remaining = _extract_dates(block)
    return facts, dates, _extract_tokens(remaining, result.currency)


def _shown(fact_value: float, unit: Unit, tokens) -> bool:
    target = fact_value * 100 if unit is Unit.RATE else fact_value
    return any(math.isclose(t.value, target, **_TIGHT) for t in tokens)


_CASES = [(engine, k, dropped) for engine in ENGINES for k, dropped in ((3, 0), (12, 7), (50, 50))]


@pytest.fixture(
    scope="module",
    params=_CASES,
    ids=[f"{engine}-k{k}-dropped{dropped}" for engine, k, dropped in _CASES],
)
def result(request: pytest.FixtureRequest) -> RiskResult:
    engine, k, dropped = request.param
    missing = min(k - 1, 49)
    return run(engine, make_portfolio(k, dropped=dropped, missing_per_date=missing))


# ------------------------------------------------ prompt <-> pool bijection


def test_every_number_in_the_portfolio_prompt_is_a_fact_in_its_own_unit(result):
    facts, _, tokens = _prompt_tokens(result)

    assert tokens
    assert [t.text for t in tokens if not _matches(t, facts, **_TIGHT)] == []


def test_every_portfolio_fact_is_shown_in_the_prompt(result):
    facts, dates, tokens = _prompt_tokens(result)

    assert [f for f in facts.numbers if not _shown(f.forms[0], f.unit, tokens)] == []
    assert set(dates) == set(facts.dates)


def test_the_portfolio_prompt_read_back_passes_the_numeric_check(result):
    verify_numeric_consistency(_facts_block(build_explanation_prompt(result)), result)


def test_the_portfolio_prompt_fits_the_context_window_with_the_output(result):
    prompt = build_explanation_prompt(result)

    assert prompt_token_bound(prompt) + config.OLLAMA_NUM_PREDICT <= config.OLLAMA_NUM_CTX


# ------------------------------------------------------ what is shown


def _positions(prompt: str) -> list[str]:
    block = _facts_block(prompt).split("Positions, largest first:\n")[1]
    return [line for line in block.split("\n") if line.startswith("- ")][: TOP_POSITIONS + 1]


def test_the_ten_largest_positions_are_listed_and_the_rest_aggregated():
    portfolio = make_portfolio(12)
    result = run("historical-VaR", portfolio)
    by_size = sorted(zip(portfolio.notionals, portfolio.asset_ids, strict=True), reverse=True)

    lines = _positions(build_explanation_prompt(result))

    assert [line.split(":")[0][2:] for line in lines[:TOP_POSITIONS]] == [
        asset_id for _, asset_id in by_size[:TOP_POSITIONS]
    ]
    rest = sum(n for n, _ in by_size[TOP_POSITIONS:])
    assert lines[TOP_POSITIONS] == (
        f"- 2 other positions together: {rest:,.2f} USD, "
        f"{rest / portfolio.total_value * 100:.2f}% of the portfolio"
    )


def test_weights_are_recomputed_from_the_notionals():
    portfolio = make_portfolio(3)
    result = run("parametric-VaR", portfolio)
    top = max(zip(portfolio.notionals, portfolio.asset_ids, strict=True))

    lines = _positions(build_explanation_prompt(result))

    assert lines[0] == (
        f"- {top[1]}: {top[0]:,.2f} USD, "
        f"{top[0] / portfolio.total_value * 100:.2f}% of the portfolio"
    )


def test_only_the_first_dropped_dates_are_listed_with_capped_missing_assets():
    result = run("historical-VaR", make_portfolio(12, dropped=7, missing_per_date=8))
    prompt = build_explanation_prompt(result)
    dropped = result.metadata["dropped_dates"]

    assert "- dates dropped because some asset had no return on them: 7" in prompt
    for day in dropped[:LISTED_DROPPED_DATES]:
        assert (
            f"- {day}: no return for ASSET001, ASSET002, ASSET003, ASSET004, ASSET005 "
            "and 3 more" in prompt
        )
    for day in dropped[LISTED_DROPPED_DATES:]:
        assert day not in prompt
    assert "- and 2 more dropped dates" in prompt


def test_nested_metadata_is_summarised_never_dumped():
    result = run("monte_carlo-ES", make_portfolio(12, dropped=7, missing_per_date=3))
    prompt = build_explanation_prompt(result)

    for raw in ("alignment_by_asset", "n_input", "dropped_dates_missing_assets", "{", "["):
        assert raw not in prompt
    assert str(result.metadata["weights"][0]) not in prompt
    assert "random_draw_layout" not in prompt and "seed" not in prompt


def test_the_prompt_does_not_grow_with_the_number_of_assets():
    small = build_explanation_prompt(run("historical-VaR", make_portfolio(12)))
    large = build_explanation_prompt(run("historical-VaR", make_portfolio(50)))

    # Same lines either way; only the figures on them differ.
    assert large.count("\n") == small.count("\n")
    assert len(large) < 1.1 * len(small)


def test_the_diagnostics_are_those_the_method_has():
    historical = build_explanation_prompt(run("historical-VaR", make_portfolio(3)))
    parametric = build_explanation_prompt(run("parametric-ES", make_portfolio(3)))
    monte_carlo = build_explanation_prompt(run("monte_carlo-VaR", make_portfolio(3)))

    assert "expected observations in the tail" in historical
    assert "condition number" not in historical
    assert "condition number" in parametric and "simulations" not in parametric
    assert "- simulations: 2,000" in monte_carlo and "condition number" in monte_carlo


def test_the_prompt_rules_contain_no_digits():
    prompt = build_explanation_prompt(run("historical-VaR", make_portfolio(3)))

    assert not any(ch.isdigit() for ch in prompt.split("\n\n")[1])
    assert not any(ch.isdigit() for ch in prompt.split("\n\n")[0])


# ------------------------------------------------ edge validation (422)


def _hand_built(**metadata_overrides) -> RiskResult:
    metadata = {
        "notionals": [600_000.0, 400_000.0],
        "weights": [0.6, 0.4],
        "dropped_dates": [],
        "dropped_dates_missing_assets": {},
        **metadata_overrides,
    }
    return RiskResult(
        method=RiskMethod.HISTORICAL,
        metric=RiskMetric.VAR,
        value=34_919.0,
        confidence_level=0.95,
        horizon_days=1,
        portfolio_value=1_000_000.0,
        as_of=date(2026, 8, 21),
        n_observations=250,
        asset_ids=["AAPL", "MSFT"],
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"notionals": None}, "metadata.notionals"),
        ({"weights": None}, "metadata.weights"),
        ({"notionals": [1_000_000.0]}, "one entry per asset_id"),
        ({"notionals": [600_000.0, "400000"]}, "finite, non-negative"),
        ({"notionals": [600_000.0, float("nan")]}, "finite, non-negative"),
        ({"weights": [0.6, -0.4]}, "finite, non-negative"),
        ({"notionals": [700_000.0, 400_000.0], "weights": [7 / 11, 4 / 11]}, "portfolio_value"),
        ({"weights": [0.5, 0.5]}, "disagree"),
        ({"dropped_dates": ["2026-02-30"]}, "dropped_dates"),
        ({"dropped_dates": "2026-01-05"}, "dropped_dates"),
        ({"dropped_dates_missing_assets": {"2026-01-05": "AAPL"}}, "missing_assets"),
    ],
)
def test_an_inconsistent_portfolio_result_is_refused(overrides, message):
    with pytest.raises(InvalidParameterError, match=message):
        build_fact_sheet(_hand_built(**overrides))


def test_duplicate_asset_ids_are_refused():
    result = _hand_built()
    result.asset_ids[1] = "AAPL"

    with pytest.raises(InvalidParameterError, match="distinct"):
        build_fact_sheet(result)


def test_the_hand_built_result_is_itself_valid():
    facts: FactSheet = build_fact_sheet(_hand_built())

    assert facts.is_portfolio


# ------------------------------------ fabricated explanations (design §5)


@pytest.mark.parametrize(
    "text",
    [
        "The 1-day VaR at 95.00% confidence is $34,919.00 on a portfolio of "
        "1,000,000.00 USD: AAPL is 600,000.00 USD (60.00%) and MSFT 400,000.00 USD (40.00%).",
        "Roughly $34,900 could be lost in a day, in 5% of cases.",
    ],
)
def test_a_sound_portfolio_explanation_passes(text):
    verify_numeric_consistency(text, _hand_built())


@pytest.mark.parametrize(
    ("text", "reported"),
    [
        ("A loss of $60 is the threshold.", "$60"),  # 60% is a weight, not money
        ("The 5-day VaR is $34,919.00.", "5-day"),  # (1 - 0.95) * 100, as a horizon
        ("The VaR is about $35,500.", "35,500"),  # 1.7% off 34,919
        ("AAPL alone accounts for $21,000 of it.", "21,000"),  # an invented split
        ("As of 2026-08-22 the VaR is $34,919.", "2026-08-22"),
    ],
)
def test_a_fabricated_portfolio_figure_is_rejected(text, reported):
    with pytest.raises(NumericConsistencyError, match=re.escape(reported)):
        verify_numeric_consistency(text, _hand_built())


def test_an_invented_contribution_equal_to_a_weight_is_caught_by_the_lexical_guard():
    # Every number reconciles (60% is AAPL's weight); only the claim is wrong.
    text = "AAPL contributes 60.00% of the $34,919.00 VaR."
    verify_numeric_consistency(text, _hand_built())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": text})

    client = OllamaClient(
        model="qwen3:8b",
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://o"),
    )
    with pytest.raises(UnsupportedClaimError, match="attribution"):
        generate_explanation(_hand_built(), client)
