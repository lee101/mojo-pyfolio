"""Transaction analytics compatible with pyfolio.txn."""

from __future__ import annotations

import pandas as pd


def get_txn_vol(transactions):
    normalized = transactions.copy()
    normalized.index = normalized.index.normalize()
    amounts = normalized.amount.abs()
    values = amounts * normalized.price
    daily_amounts = amounts.groupby(amounts.index).sum()
    daily_values = values.groupby(values.index).sum()
    daily_amounts.name = "txn_shares"
    daily_values.name = "txn_volume"
    return pd.concat([daily_values, daily_amounts], axis=1)


def adjust_returns_for_slippage(
    returns, positions, transactions, slippage_bps
):
    slippage = 0.0001 * slippage_bps
    portfolio_value = positions.sum(axis=1)
    pnl = portfolio_value * returns
    slippage_dollars = get_txn_vol(transactions).txn_volume * slippage
    adjusted_pnl = pnl.add(-slippage_dollars, fill_value=0)
    return returns * adjusted_pnl / pnl


def get_turnover(positions, transactions, denominator="AGB"):
    traded_value = get_txn_vol(transactions).txn_volume
    if denominator == "AGB":
        actual_gross_book = positions.drop("cash", axis=1).abs().sum(axis=1)
        denom = actual_gross_book.rolling(2).mean()
        denom.iloc[0] = actual_gross_book.iloc[0] / 2
    elif denominator == "portfolio_value":
        denom = positions.sum(axis=1)
    else:
        raise ValueError(
            f"Unexpected value for denominator '{denominator}'. The denominator "
            "parameter must be either 'AGB' or 'portfolio_value'."
        )
    denom.index = denom.index.normalize()
    return traded_value.div(denom, axis="index").fillna(0)
