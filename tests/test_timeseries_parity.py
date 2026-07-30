"""Numerical and behavioural parity with pyfolio-reloaded 0.9.7."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pyfolio import timeseries as upstream

from mojopyfolio import timeseries as mojo
from mojopyfolio._lib import central_moments

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2018-01-02", periods=1_500)
    values = rng.normal(0.00035, 0.012, index.size)
    values[17] = np.nan
    values[801] = np.nan
    return pd.Series(values, index=index, name="strategy")


@pytest.fixture(scope="module")
def factor(returns):
    rng = np.random.default_rng(8)
    values = 0.25 * returns.fillna(0).to_numpy() + rng.normal(
        0.0001, 0.008, len(returns)
    )
    values[300] = np.nan
    return pd.Series(values, index=returns.index, name="benchmark")


@pytest.mark.parametrize(
    "name",
    [
        "max_drawdown",
        "annual_return",
        "annual_volatility",
        "calmar_ratio",
        "omega_ratio",
        "sortino_ratio",
        "downside_risk",
        "sharpe_ratio",
        "tail_ratio",
        "common_sense_ratio",
    ],
)
def test_scalar_risk_metrics_match_upstream(returns, name):
    actual = getattr(mojo, name)(returns)
    expected = getattr(upstream, name)(returns)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_stability_matches_upstream(returns):
    assert mojo.stability_of_timeseries(returns) == pytest.approx(
        upstream.stability_of_timeseries(returns), abs=2e-10
    )


@pytest.mark.parametrize("period", ["daily", "weekly", "monthly"])
def test_period_aware_metrics_match_upstream(returns, period):
    for name in ("annual_return", "annual_volatility", "calmar_ratio"):
        assert getattr(mojo, name)(returns, period=period) == pytest.approx(
            getattr(upstream, name)(returns, period=period)
        )
    assert mojo.sharpe_ratio(returns, risk_free=0.0001, period=period) == pytest.approx(
        upstream.sharpe_ratio(returns, risk_free=0.0001, period=period)
    )
    assert mojo.downside_risk(
        returns, required_return=0.0002, period=period
    ) == pytest.approx(
        upstream.downside_risk(
            returns, required_return=0.0002, period=period
        )
    )


def test_sortino_preserves_upstream_period_quirk(returns):
    assert mojo.sortino_ratio(
        returns, required_return=0.0002, period="monthly"
    ) == pytest.approx(
        upstream.sortino_ratio(
            returns, required_return=0.0002, period="monthly"
        )
    )


def test_cumulative_returns_series_matches_upstream(returns):
    for starting_value in (0, 1, 100):
        actual = mojo.cum_returns(returns, starting_value)
        expected = upstream.cum_returns(returns, starting_value)
        pd.testing.assert_series_equal(actual, expected)


def test_cumulative_returns_dataframe_matches_upstream(returns, factor):
    frame = pd.concat([returns, factor], axis=1)
    pd.testing.assert_frame_equal(
        mojo.cum_returns(frame, 1), upstream.cum_returns(frame, 1)
    )


def test_factor_metrics_match_upstream(returns, factor):
    assert mojo.alpha(returns, factor) == pytest.approx(upstream.alpha(returns, factor))
    assert mojo.beta(returns, factor) == pytest.approx(upstream.beta(returns, factor))
    assert mojo.alpha_beta(returns, factor) == pytest.approx(
        upstream.alpha_beta(returns, factor)
    )


def test_factor_metrics_align_mismatched_indices(returns, factor):
    shifted = factor.iloc[50:].copy()
    shifted.index = shifted.index.shift(1, freq="B")
    assert mojo.alpha(returns, shifted) == pytest.approx(
        upstream.alpha(returns, shifted)
    )
    assert mojo.beta(returns, shifted) == pytest.approx(
        upstream.beta(returns, shifted)
    )


def test_rolling_metrics_match_upstream(returns, factor):
    pd.testing.assert_series_equal(
        mojo.rolling_volatility(returns, 63),
        upstream.rolling_volatility(returns, 63),
        rtol=2e-12,
        atol=2e-12,
    )
    pd.testing.assert_series_equal(
        mojo.rolling_sharpe(returns, 63),
        upstream.rolling_sharpe(returns, 63),
        rtol=2e-12,
        atol=2e-12,
    )
    pd.testing.assert_series_equal(
        mojo.rolling_beta(returns, factor, 63),
        upstream.rolling_beta(returns, factor, 63),
        check_names=False,
        rtol=2e-12,
        atol=2e-12,
    )


def test_rolling_beta_dataframe_matches_upstream(returns, factor):
    factors = pd.DataFrame({"market": factor, "alt": factor * 0.5 + 0.001})
    actual = mojo.rolling_beta(returns, factors, 40)
    expected = upstream.rolling_beta(returns, factors, 40)
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("convert_to", ["weekly", "monthly", "yearly"])
def test_aggregate_returns_matches_upstream(returns, convert_to):
    pd.testing.assert_series_equal(
        mojo.aggregate_returns(returns, convert_to),
        upstream.aggregate_returns(returns, convert_to),
    )


def test_miscellaneous_metrics_match_upstream(returns):
    assert mojo.var_cov_var_normal(1_000_000, 0.95, sigma=0.02) == pytest.approx(
        upstream.var_cov_var_normal(1_000_000, 0.95, sigma=0.02)
    )
    assert mojo.value_at_risk(returns) == pytest.approx(
        upstream.value_at_risk(returns)
    )
    assert mojo.value_at_risk(returns, "monthly", 1.5) == pytest.approx(
        upstream.value_at_risk(returns, "monthly", 1.5)
    )
    pd.testing.assert_series_equal(
        mojo.normalize(returns.fillna(0), 100),
        upstream.normalize(returns.fillna(0), 100),
    )


def test_perf_stats_matches_upstream(returns, factor):
    actual = mojo.perf_stats(returns, factor_returns=factor)
    expected = upstream.perf_stats(returns, factor_returns=factor)
    pd.testing.assert_series_equal(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("size", [7, 1_048_579])
def test_central_moment_simd_tail_and_parallel_threshold(size):
    values = np.random.default_rng(71).normal(0.0003, 0.012, size)
    mean, second, third, fourth = central_moments(values)
    centered = values - np.mean(values)
    assert mean == pytest.approx(np.mean(values), rel=2e-13, abs=2e-13)
    assert second == pytest.approx(np.mean(centered**2), rel=2e-13, abs=2e-13)
    assert third == pytest.approx(np.mean(centered**3), rel=2e-13, abs=2e-13)
    assert fourth == pytest.approx(np.mean(centered**4), rel=2e-13, abs=2e-13)


@pytest.mark.filterwarnings("ignore:Precision loss occurred")
def test_perf_stats_constant_moments_match_upstream():
    returns = pd.Series(np.full(500, 0.001))
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = mojo.perf_stats(returns)
        expected = upstream.perf_stats(returns)
    assert actual["Skew"] == pytest.approx(expected["Skew"])
    assert actual["Kurtosis"] == pytest.approx(expected["Kurtosis"])


def test_drawdown_periods_and_table_match_upstream(returns):
    clean = returns.fillna(0)
    assert mojo.get_max_drawdown(clean) == upstream.get_max_drawdown(clean)
    assert mojo.get_top_drawdowns(clean, 5) == upstream.get_top_drawdowns(clean, 5)
    actual = mojo.gen_drawdown_table(clean, 5)
    expected = upstream.gen_drawdown_table(clean, 5)
    pd.testing.assert_frame_equal(actual, expected)


def test_distribution_stats_matches_upstream():
    values = np.array([-3.0, -1.0, 0.5, 2.0, 8.0])
    pd.testing.assert_series_equal(
        mojo.calc_distribution_stats(values),
        upstream.calc_distribution_stats(values),
    )


def test_bootstrap_matches_upstream_with_same_seed(returns):
    clean = returns.dropna()
    np.random.seed(123)
    actual = mojo.calc_bootstrap(mojo.sharpe_ratio, clean, n_samples=20)
    np.random.seed(123)
    expected = upstream.calc_bootstrap(
        upstream.sharpe_ratio, clean, n_samples=20
    )
    assert actual == pytest.approx(expected)


def test_simulate_and_summarize_paths_match_upstream(returns):
    clean = returns.dropna()
    actual_samples = mojo.simulate_paths(
        clean, 80, num_samples=200, random_seed=9
    )
    expected_samples = upstream.simulate_paths(
        clean, 80, num_samples=200, random_seed=9
    )
    assert np.array_equal(actual_samples, expected_samples)
    pd.testing.assert_frame_equal(
        mojo.summarize_paths(actual_samples, cone_std=(1, 2), starting_value=10),
        upstream.summarize_paths(
            expected_samples, cone_std=(1, 2), starting_value=10
        ),
        rtol=2e-12,
        atol=2e-12,
    )


def test_forecast_cone_matches_upstream(returns):
    clean = returns.dropna()
    actual = mojo.forecast_cone_bootstrap(
        clean, 50, cone_std=1.5, num_samples=100, random_seed=6
    )
    expected = upstream.forecast_cone_bootstrap(
        clean, 50, cone_std=1.5, num_samples=100, random_seed=6
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-12, atol=2e-12)


def test_perf_stats_bootstrap_matches_upstream(returns):
    clean = returns.dropna().iloc[:8]
    np.random.seed(17)
    actual = mojo.perf_stats_bootstrap(clean)
    np.random.seed(17)
    expected = upstream.perf_stats_bootstrap(clean)
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-10, atol=2e-10)


def test_gross_leverage_matches_upstream():
    positions = pd.DataFrame(
        {"asset": [50.0, -25.0], "cash": [50.0, 125.0]}
    )
    pd.testing.assert_series_equal(
        mojo.gross_lev(positions), upstream.gross_lev(positions)
    )


def test_ffi_shape_and_empty_guards():
    with pytest.raises(ValueError, match="one-dimensional"):
        mojo.alpha(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="equal length"):
        mojo.beta(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="positive integer"):
        mojo.rolling_beta(np.ones(3), np.ones(3), 0)
    with pytest.raises(ValueError, match="at least 2"):
        mojo.rolling_volatility(np.ones(3), 1)
    with pytest.raises(ValueError, match="at least one path"):
        mojo.summarize_paths(np.empty((0, 3)))
    assert mojo.summarize_paths(np.empty((2, 0))).empty


def test_empty_and_constant_edges():
    empty = pd.Series([], dtype=float)
    assert np.isnan(mojo.annual_return(empty))
    assert np.isnan(mojo.max_drawdown(empty))
    assert np.isnan(mojo.sharpe_ratio([0.01]))
    assert np.isnan(mojo.calmar_ratio(pd.Series(np.zeros(100))))
