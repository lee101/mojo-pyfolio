"""Position analytics compatible with pyfolio.pos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_percent_alloc(values):
    return values.divide(values.sum(axis="columns"), axis="rows")


def get_top_long_short_abs(positions, top=10):
    positions = positions.drop("cash", axis="columns")
    maximum = positions.max()
    minimum = positions.min()
    absolute = positions.abs().max()
    return (
        maximum[maximum > 0].nlargest(top),
        minimum[minimum < 0].nsmallest(top),
        absolute.nlargest(top),
    )


def get_max_median_position_concentration(positions):
    exposure = get_percent_alloc(positions).drop("cash", axis=1)
    longs = exposure.where(exposure > 0)
    shorts = exposure.where(exposure < 0)
    return pd.DataFrame(
        {
            "max_long": longs.max(axis=1),
            "median_long": longs.median(axis=1),
            "median_short": shorts.median(axis=1),
            "max_short": shorts.min(axis=1),
        }
    )


def get_sector_exposures(positions, symbol_sector_map):
    cash = positions["cash"]
    securities = positions.drop("cash", axis=1)
    mapped = {
        column: symbol_sector_map[column]
        for column in securities.columns
        if column in symbol_sector_map
    }
    result = securities[list(mapped)].T.groupby(mapped).sum().T
    result["cash"] = cash
    return result


def get_long_short_pos(positions):
    securities = positions.drop("cash", axis=1)
    longs = securities[securities > 0].sum(axis=1).fillna(0)
    shorts = securities[securities < 0].sum(axis=1).fillna(0)
    net_liquidation = longs + shorts + positions.cash
    result = pd.DataFrame(
        {
            "long": longs / net_liquidation,
            "short": shorts / net_liquidation,
        }
    )
    result["net exposure"] = result["long"] + result["short"]
    return result
