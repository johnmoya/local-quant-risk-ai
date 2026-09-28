"""Refresh the raw price CSV used by the real-data validation study.

Running this is **optional**. `data/research/SPY_prices.csv` is committed,
and it is the reproducible source of truth: the study reads that file and
needs neither this script, the `research` extra, nor a network connection.
This exists so the window can be extended or the instrument changed later,
not as a step in reproducing the published results — an external API that
still answers today is exactly the kind of dependency this project has
already been bitten by (unpinned Docker tags, inferred date formats).

Usage:
    uv run --extra research python -m research.download_data

The file is written in the two-column ISO-8601 shape
`quant_risk_ai.data.loaders.load_price_series` reads by default, so the
study ingests it through the engine's own loader rather than a bespoke
reader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TICKER = "SPY"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_OUTPUT = Path("data/research/SPY_prices.csv")


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a daily adjusted close series as a `date,price` frame.

    `auto_adjust=True` gives a total-return series: prices adjusted for
    dividends and splits. Unadjusted closes would show an artificial drop
    on every ex-dividend date — for a quarterly payer like SPY that is
    roughly 0.3-0.4% four times a year, injected straight into the return
    series the VaR models are estimated from.
    """
    import yfinance as yf  # imported lazily: research extra, not a runtime dep

    frame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
    )
    if frame is None or frame.empty:
        raise SystemExit(
            f"no data returned for {ticker!r} between {start} and {end}. "
            f"Yahoo Finance is an unofficial, unauthenticated endpoint and does "
            f"rate-limit; the committed CSV remains the reproducible source."
        )

    prices = frame["Close"].dropna()
    return pd.DataFrame({"date": prices.index.strftime("%Y-%m-%d"), "price": prices.to_numpy()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = download_prices(args.ticker, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"wrote {len(frame)} rows to {args.output}")
    print(f"range: {frame['date'].iloc[0]} .. {frame['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
