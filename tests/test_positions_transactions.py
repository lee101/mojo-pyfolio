"""Parity for the non-kernel portfolio table helpers."""

import numpy as np
import pandas as pd

from pyfolio import pos as upstream_pos
from pyfolio import txn as upstream_txn

from mojopyfolio import pos, txn


def position_frame():
    index = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "A": [100, 120, -50, 80, 30],
            "B": [-40, -20, 90, 10, -60],
            "C": [10, 20, 30, -40, -10],
            "cash": [1000, 980, 1030, 950, 1040],
        },
        index=index,
        dtype=float,
    )


def transaction_frame():
    index = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-01 14:00",
            "2024-01-03 11:00",
            "2024-01-05 15:00",
        ]
    )
    return pd.DataFrame(
        {
            "amount": [10, -3, 7, -5],
            "price": [100.0, 101.0, 50.0, 80.0],
        },
        index=index,
    )


def test_percent_alloc_and_top_positions_match_upstream():
    positions = position_frame()
    pd.testing.assert_frame_equal(
        pos.get_percent_alloc(positions),
        upstream_pos.get_percent_alloc(positions),
    )
    actual = pos.get_top_long_short_abs(positions, 2)
    expected = upstream_pos.get_top_long_short_abs(positions, 2)
    for left, right in zip(actual, expected):
        pd.testing.assert_series_equal(left, right)


def test_long_short_positions_match_upstream():
    positions = position_frame()
    pd.testing.assert_frame_equal(
        pos.get_long_short_pos(positions),
        upstream_pos.get_long_short_pos(positions),
    )


def test_position_concentration_reference():
    positions = position_frame()
    exposure = positions.div(positions.sum(axis=1), axis=0).drop(columns="cash")
    expected = pd.DataFrame(
        {
            "max_long": exposure.where(exposure > 0).max(axis=1),
            "median_long": exposure.where(exposure > 0).median(axis=1),
            "median_short": exposure.where(exposure < 0).median(axis=1),
            "max_short": exposure.where(exposure < 0).min(axis=1),
        }
    )
    pd.testing.assert_frame_equal(
        pos.get_max_median_position_concentration(positions), expected
    )


def test_sector_exposure_reference():
    positions = position_frame()
    expected = pd.DataFrame(
        {
            "defensive": positions["C"],
            "growth": positions["A"] + positions["B"],
            "cash": positions["cash"],
        }
    )
    pd.testing.assert_frame_equal(
        pos.get_sector_exposures(
            positions, {"A": "growth", "B": "growth", "C": "defensive"}
        ),
        expected,
    )


def test_transaction_volume_matches_upstream():
    transactions = transaction_frame()
    pd.testing.assert_frame_equal(
        txn.get_txn_vol(transactions),
        upstream_txn.get_txn_vol(transactions),
    )


def test_turnover_matches_upstream():
    positions = position_frame()
    transactions = transaction_frame()
    for denominator in ("AGB", "portfolio_value"):
        pd.testing.assert_series_equal(
            txn.get_turnover(positions, transactions, denominator),
            upstream_txn.get_turnover(positions, transactions, denominator),
        )


def test_slippage_adjustment_matches_upstream():
    positions = position_frame()
    transactions = transaction_frame()
    returns = pd.Series(
        [0.01, -0.005, 0.002, 0.003, -0.004], index=positions.index
    )
    pd.testing.assert_series_equal(
        txn.adjust_returns_for_slippage(returns, positions, transactions, 5),
        upstream_txn.adjust_returns_for_slippage(
            returns, positions, transactions, 5
        ),
    )
