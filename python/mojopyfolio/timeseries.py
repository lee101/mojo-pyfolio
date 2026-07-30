"""pyfolio-compatible performance and risk analytics."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ._lib import (
    address,
    central_moments,
    f64,
    factor_summary,
    lib,
    max_drawdown_value,
    summary,
)
from .txn import get_turnover

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252
DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
YEARLY = "yearly"
ANNUALIZATION_FACTORS = {DAILY: 252, WEEKLY: 52, MONTHLY: 12, YEARLY: 1}


def _annualization(period):
    try:
        return ANNUALIZATION_FACTORS[period]
    except KeyError as exc:
        raise ValueError(f"Period cannot be '{period}'.") from exc


def _values(data):
    return f64(np.asarray(data))


def _wrap_like(data, values):
    if isinstance(data, pd.Series):
        return pd.Series(values, index=data.index, name=data.name, copy=False)
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(
            values, index=data.index, columns=data.columns, copy=False
        )
    return values


def _columns(data, function):
    array = np.asarray(data)
    if array.ndim == 1:
        return function(array)
    values = np.array([function(array[:, column]) for column in range(array.shape[1])])
    if isinstance(data, pd.DataFrame):
        return pd.Series(values, index=data.columns)
    return values


def _aligned(returns, factor_returns):
    if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
        if returns.ndim != 1 or factor_returns.ndim != 1:
            raise ValueError("returns and factor_returns must be one-dimensional")
        if returns.size != factor_returns.size:
            raise ValueError("returns and factor_returns must have equal length")
        return f64(returns), f64(factor_returns)
    frame = pd.concat([pd.Series(returns), pd.Series(factor_returns)], axis=1)
    return f64(frame.iloc[:, 0]), f64(frame.iloc[:, 1])


def var_cov_var_normal(P, c, mu=0, sigma=1):
    alpha_value = scipy_stats.norm.ppf(1 - c, mu, sigma)
    return P - P * (alpha_value + 1)


def max_drawdown(returns):
    return _columns(
        returns,
        lambda values: np.nan if len(values) == 0 else max_drawdown_value(values),
    )


def annual_return(returns, period=DAILY):
    annualization = _annualization(period)

    def calculate(values):
        if len(values) < 1:
            return np.nan
        ending_value = summary(values)[6]
        return ending_value ** (annualization / len(values)) - 1

    return _columns(returns, calculate)


def annual_volatility(returns, period=DAILY):
    annualization = _annualization(period)

    def calculate(values):
        result = summary(values)
        count = result[0]
        if len(values) < 2 or count < 2:
            return np.nan
        variance = (result[2] - result[1] ** 2 / count) / (count - 1)
        return np.sqrt(max(0.0, variance)) * np.sqrt(annualization)

    return _columns(returns, calculate)


def calmar_ratio(returns, period=DAILY):
    drawdown = max_drawdown(returns)
    if np.ndim(drawdown):
        annual = annual_return(returns, period)
        return pd.Series(
            np.where(drawdown < 0, annual / np.abs(drawdown), np.nan),
            index=getattr(drawdown, "index", None),
        )
    if drawdown >= 0:
        return np.nan
    value = annual_return(returns, period) / abs(drawdown)
    return np.nan if np.isinf(value) else value


def omega_ratio(returns, annual_return_threshhold=0.0):
    if annual_return_threshhold <= -1:
        return np.nan
    threshold = (1 + annual_return_threshhold) ** (1 / 252) - 1

    def calculate(values):
        if len(values) < 2:
            return np.nan
        result = summary(values, omega_offset=threshold)
        return result[4] / result[5] if result[5] > 0 else np.nan

    return _columns(returns, calculate)


def downside_risk(returns, required_return=0, period=DAILY):
    annualization = _annualization(period)

    def calculate(values):
        if len(values) < 1:
            return np.nan
        result = summary(values, required_return=float(required_return))
        if result[0] == 0:
            return np.nan
        return np.sqrt(result[3] / result[0]) * np.sqrt(annualization)

    return _columns(returns, calculate)


def sortino_ratio(returns, required_return=0, period=DAILY):
    # Upstream's wrapper accepts period but delegates without forwarding it.
    def calculate(values):
        if len(values) < 2:
            return np.nan
        result = summary(values, required_return=float(required_return))
        risk = np.sqrt(result[3] / result[0]) * np.sqrt(252)
        return ((result[1] / result[0] - required_return) * 252) / risk

    return _columns(returns, calculate)


def sharpe_ratio(returns, risk_free=0, period=DAILY):
    annualization = _annualization(period)

    def calculate(values):
        result = summary(values)
        count = result[0]
        if len(values) < 2 or count < 2:
            return np.nan
        variance = (result[2] - result[1] ** 2 / count) / (count - 1)
        deviation = np.sqrt(max(0.0, variance))
        return (result[1] / count - risk_free) / deviation * np.sqrt(annualization)

    return _columns(returns, calculate)


def _alpha_beta_values(returns, factor_returns):
    left, right = _aligned(returns, factor_returns)
    result = factor_summary(left, right)
    count, sum_y, sum_x, sum_xx, sum_xy = result
    if count < 2:
        return np.nan, np.nan
    denominator = sum_xx - sum_x * sum_x / count
    beta_value = (
        np.nan if denominator < 1e-30 else (sum_xy - sum_x * sum_y / count) / denominator
    )
    alpha_mean = (sum_y - beta_value * sum_x) / count
    alpha_value = (1 + alpha_mean) ** 252 - 1
    return alpha_value, beta_value


def alpha_beta(returns, factor_returns):
    return np.asarray(_alpha_beta_values(returns, factor_returns))


def alpha(returns, factor_returns):
    return _alpha_beta_values(returns, factor_returns)[0]


def beta(returns, factor_returns):
    return _alpha_beta_values(returns, factor_returns)[1]


def stability_of_timeseries(returns):
    def calculate(values):
        result = summary(values)
        count = result[0]
        if len(values) < 2 or count < 2:
            return np.nan
        sum_x = count * (count - 1) / 2
        sum_xx = count * (count - 1) * (2 * count - 1) / 6
        numerator = count * result[10] - sum_x * result[8]
        denominator = np.sqrt(
            (count * sum_xx - sum_x**2)
            * (count * result[9] - result[8] ** 2)
        )
        correlation = numerator / denominator
        return correlation * correlation

    return _columns(returns, calculate)


def tail_ratio(returns):
    def calculate(values):
        clean = values[~np.isnan(values)]
        if clean.size == 0:
            return np.nan
        upper, lower = np.percentile(clean, (95, 5))
        return abs(upper) / abs(lower)

    return _columns(returns, calculate)


def common_sense_ratio(returns):
    return tail_ratio(returns) * (1 + annual_return(returns))


def normalize(returns, starting_value=1):
    return starting_value * (returns / returns.iloc[0])


def cum_returns(returns, starting_value=0):
    source = _values(returns)
    if source.size == 0:
        return returns.copy()
    if source.ndim == 1:
        rows, columns = source.shape[0], 1
    elif source.ndim == 2:
        rows, columns = source.shape
    else:
        raise ValueError("returns must be one- or two-dimensional")
    destination = np.empty_like(source)
    lib().mpf_cum_returns(
        address(source), address(destination), rows, columns, float(starting_value)
    )
    if isinstance(returns, pd.Series):
        return pd.Series(destination, index=returns.index, copy=False)
    return _wrap_like(returns, destination)


def cum_returns_final(returns, starting_value=0):
    return _columns(
        returns,
        lambda values: (
            np.nan
            if len(values) == 0
            else (summary(values)[6] - 1 if starting_value == 0 else summary(values)[6] * starting_value)
        ),
    )


def drawdown_series(returns):
    source = f64(returns).ravel()
    destination = np.empty_like(source)
    if source.size:
        lib().mpf_drawdown_series(
            address(source), address(destination), source.size
        )
    return _wrap_like(returns, destination)


def aggregate_returns(returns, convert_to):
    def cumulate(group):
        return cum_returns(group).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda value: value.year, lambda value: value.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda value: value.year, lambda value: value.month]
    elif convert_to == YEARLY:
        grouping = [lambda value: value.year]
    else:
        raise ValueError("convert_to must be weekly, monthly or yearly")
    return returns.groupby(grouping).apply(cumulate)


def rolling_beta(
    returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6
):
    if not isinstance(rolling_window, (int, np.integer)) or rolling_window < 1:
        raise ValueError("rolling_window must be a positive integer")
    if getattr(factor_returns, "ndim", 1) > 1:
        return factor_returns.apply(
            lambda column: rolling_beta(returns, column, rolling_window)
        )
    left, right = _aligned(returns, factor_returns)
    destination = np.full(left.size, np.nan)
    if left.size:
        lib().mpf_rolling_beta(
            address(left), address(right), address(destination), left.size, rolling_window
        )
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(left.size)
    return pd.Series(destination, index=index)


def gross_lev(positions):
    exposure = positions.drop("cash", axis=1).abs().sum(axis=1)
    return exposure / positions.sum(axis=1)


def value_at_risk(returns, period=None, sigma=2.0):
    returns_agg = aggregate_returns(returns, period) if period is not None else returns.copy()
    return returns_agg.mean() - sigma * returns_agg.std()


SIMPLE_STAT_FUNCS = [
    annual_return,
    cum_returns_final,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    scipy_stats.skew,
    scipy_stats.kurtosis,
    tail_ratio,
    value_at_risk,
]
FACTOR_STAT_FUNCS = [alpha, beta]
STAT_FUNC_NAMES = {
    "annual_return": "Annual return",
    "cum_returns_final": "Cumulative returns",
    "annual_volatility": "Annual volatility",
    "sharpe_ratio": "Sharpe ratio",
    "calmar_ratio": "Calmar ratio",
    "stability_of_timeseries": "Stability",
    "max_drawdown": "Max drawdown",
    "omega_ratio": "Omega ratio",
    "sortino_ratio": "Sortino ratio",
    "skew": "Skew",
    "kurtosis": "Kurtosis",
    "tail_ratio": "Tail ratio",
    "value_at_risk": "Daily value at risk",
    "alpha": "Alpha",
    "beta": "Beta",
}


def perf_stats(
    returns,
    factor_returns=None,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
):
    source = f64(returns).ravel()
    fused = summary(source)
    count = fused[0]
    total_count = len(source)
    annual = fused[6] ** (252 / total_count) - 1 if total_count else np.nan
    variance = (
        (fused[2] - fused[1] ** 2 / count) / (count - 1)
        if count >= 2
        else np.nan
    )
    deviation = np.sqrt(max(0.0, variance))
    annual_deviation = deviation * np.sqrt(252)
    sharpe = fused[1] / count / deviation * np.sqrt(252)
    downside = np.sqrt(fused[3] / count) * np.sqrt(252)
    sortino = (fused[1] / count * 252) / downside
    sum_x = count * (count - 1) / 2
    sum_xx = count * (count - 1) * (2 * count - 1) / 6
    stability_denominator = np.sqrt(
        (count * sum_xx - sum_x**2)
        * (count * fused[9] - fused[8] ** 2)
    )
    stability_correlation = (
        (count * fused[10] - sum_x * fused[8]) / stability_denominator
    )
    if total_count and count == total_count:
        mean, second_moment, third_moment, fourth_moment = central_moments(source)
        if second_moment <= (np.finfo(np.float64).eps * mean) ** 2:
            skew = np.nan
            kurtosis = np.nan
        else:
            skew = third_moment / second_moment**1.5
            kurtosis = fourth_moment / second_moment**2 - 3.0
        upper, lower = np.percentile(source, (95, 5))
        tail = abs(upper) / abs(lower)
    else:
        skew = scipy_stats.skew(source)
        kurtosis = scipy_stats.kurtosis(source)
        tail = tail_ratio(source)
    values = pd.Series(
        {
            "Annual return": annual,
            "Cumulative returns": fused[6] - 1 if total_count else np.nan,
            "Annual volatility": annual_deviation,
            "Sharpe ratio": sharpe,
            "Calmar ratio": annual / abs(fused[7]) if fused[7] < 0 else np.nan,
            "Stability": stability_correlation**2,
            "Max drawdown": fused[7],
            "Omega ratio": fused[4] / fused[5] if fused[5] > 0 else np.nan,
            "Sortino ratio": sortino,
            "Skew": skew,
            "Kurtosis": kurtosis,
            "Tail ratio": tail,
            "Daily value at risk": fused[1] / count - 2 * deviation,
        },
        dtype="float64",
    )
    if positions is not None and not positions.empty:
        values["Gross leverage"] = gross_lev(positions).mean()
        if transactions is not None and not transactions.empty:
            values["Daily turnover"] = get_turnover(
                positions, transactions, turnover_denom
            ).mean()
    if factor_returns is not None:
        for function in FACTOR_STAT_FUNCS:
            values[STAT_FUNC_NAMES[function.__name__]] = function(
                returns, factor_returns
            )
    return values


def calc_bootstrap(func, returns, *args, **kwargs):
    n_samples = kwargs.pop("n_samples", 1000)
    destination = np.empty(n_samples)
    factor_returns = kwargs.pop("factor_returns", None)
    for index in range(n_samples):
        selection = np.random.randint(len(returns), size=len(returns))
        returns_sample = returns.iloc[selection].reset_index(drop=True)
        if factor_returns is None:
            destination[index] = func(returns_sample, *args, **kwargs)
        else:
            factor_sample = factor_returns.iloc[selection].reset_index(drop=True)
            destination[index] = func(
                returns_sample, factor_sample, *args, **kwargs
            )
    return destination


def calc_distribution_stats(x):
    return pd.Series(
        {
            "mean": np.mean(x),
            "median": np.median(x),
            "std": np.std(x),
            "5%": np.percentile(x, 5),
            "25%": np.percentile(x, 25),
            "75%": np.percentile(x, 75),
            "95%": np.percentile(x, 95),
            "IQR": np.subtract.reduce(np.percentile(x, [75, 25])),
        }
    )


def perf_stats_bootstrap(returns, factor_returns=None, return_stats=True, **kwargs):
    bootstrap_values = OrderedDict()
    for function in SIMPLE_STAT_FUNCS:
        name = STAT_FUNC_NAMES[function.__name__]
        # pyfolio accepts **kwargs here but does not forward them.
        bootstrap_values[name] = calc_bootstrap(function, returns)
    if factor_returns is not None:
        for function in FACTOR_STAT_FUNCS:
            name = STAT_FUNC_NAMES[function.__name__]
            bootstrap_values[name] = calc_bootstrap(
                function,
                returns,
                factor_returns=factor_returns,
            )
    frame = pd.DataFrame(bootstrap_values)
    if return_stats:
        return frame.apply(calc_distribution_stats).T[
            ["mean", "median", "5%", "95%"]
        ]
    return frame


def get_max_drawdown_underwater(underwater):
    valley = underwater.idxmin()
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan
    return peak, valley, recovery


def get_max_drawdown(returns):
    return get_max_drawdown_underwater(drawdown_series(returns.copy()))


def get_top_drawdowns(returns, top=10):
    underwater = drawdown_series(returns.copy())
    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        if not pd.isnull(recovery):
            underwater = underwater.drop(underwater[peak:recovery].index[1:-1])
        else:
            underwater = underwater.loc[:peak]
        drawdowns.append((peak, valley, recovery))
        if len(returns) == 0 or len(underwater) == 0 or np.min(underwater) == 0:
            break
    return drawdowns


def gen_drawdown_table(returns, top=10):
    cumulative = cum_returns(returns, 1.0)
    periods = get_top_drawdowns(returns, top=top)
    frame = pd.DataFrame(
        index=range(top),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Duration",
        ],
    )
    for index, (peak, valley, recovery) in enumerate(periods):
        frame.loc[index, "Duration"] = (
            np.nan if pd.isnull(recovery) else len(pd.date_range(peak, recovery, freq="B"))
        )
        frame.loc[index, "Peak date"] = peak
        frame.loc[index, "Valley date"] = valley
        frame.loc[index, "Recovery date"] = recovery
        frame.loc[index, "Net drawdown in %"] = (
            (cumulative.loc[peak] - cumulative.loc[valley]) / cumulative.loc[peak] * 100
        )
    for column in ("Peak date", "Valley date", "Recovery date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _rolling_stat(returns, window, mode):
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("rolling window must be an integer of at least 2")
    source = f64(returns).ravel()
    destination = np.full(source.size, np.nan)
    if source.size:
        lib().mpf_rolling_stats(
            address(source),
            address(destination),
            address(destination),
            source.size,
            window,
            252.0,
            mode,
        )
    return _wrap_like(returns, destination)


def rolling_volatility(returns, rolling_vol_window):
    return _rolling_stat(returns, rolling_vol_window, 1)


def rolling_sharpe(returns, rolling_sharpe_window):
    return _rolling_stat(returns, rolling_sharpe_window, 2)


def simulate_paths(
    is_returns, num_days, starting_value=1, num_samples=1000, random_seed=None
):
    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for index in range(num_samples):
        samples[index, :] = is_returns.sample(
            num_days, replace=True, random_state=seed
        )
    return samples


def summarize_paths(samples, cone_std=(1.0, 1.5, 2.0), starting_value=1.0):
    source = f64(samples)
    if source.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    if source.shape[0] == 0:
        raise ValueError("samples must contain at least one path")
    mean = np.empty(source.shape[1])
    deviation = np.empty(source.shape[1])
    if source.shape[1]:
        lib().mpf_summarize_paths(
            address(source),
            address(mean),
            address(deviation),
            source.shape[0],
            source.shape[1],
            float(starting_value),
        )
    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]
    bounds = pd.DataFrame(
        index=range(source.shape[1]), columns=pd.Index([], dtype="float64")
    )
    for number in cone_std:
        bounds[float(number)] = mean + deviation * number
        bounds[float(-number)] = mean - deviation * number
    return bounds


def forecast_cone_bootstrap(
    is_returns,
    num_days,
    cone_std=(1.0, 1.5, 2.0),
    starting_value=1,
    num_samples=1000,
    random_seed=None,
):
    samples = simulate_paths(
        is_returns,
        num_days,
        starting_value=starting_value,
        num_samples=num_samples,
        random_seed=random_seed,
    )
    return summarize_paths(samples, cone_std=cone_std, starting_value=starting_value)
