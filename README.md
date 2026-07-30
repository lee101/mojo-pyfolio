# mojo-pyfolio

Portfolio performance and risk analytics implemented with compiled
[Mojo](https://www.modular.com/mojo) kernels and a pandas-compatible Python API.
It covers the numerical core of pyfolio: cumulative wealth and drawdown paths,
annualized risk/return ratios, factor alpha and beta, rolling analytics,
performance summaries, bootstrap forecast cones, position concentration, and
transaction turnover.

The public functions keep pyfolio's names, signatures, labeled pandas outputs,
NaN behavior, and even its documented `sortino_ratio` period quirk. For the
covered subset, changing the import is sufficient:

```python
from mojopyfolio import timeseries as pf
```

## Install

The repository pins the Mojo nightly used to build the shared library and
installs the maintained pyfolio package for parity testing:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-pyfolio.so`. Python can also build it on first
use if it is absent. Set `MOJOPYFOLIO_LIB` to load a prebuilt library from
another location.

## Usage

```python
import numpy as np
import pandas as pd
from mojopyfolio import timeseries as pf

rng = np.random.default_rng(7)
dates = pd.bdate_range("2020-01-02", periods=1_000)
returns = pd.Series(rng.normal(0.0004, 0.012, len(dates)), index=dates)
market = pd.Series(rng.normal(0.0003, 0.010, len(dates)), index=dates)

stats = pf.perf_stats(returns, factor_returns=market)
drawdowns = pf.gen_drawdown_table(returns, top=5)
rolling = pf.rolling_sharpe(returns, rolling_sharpe_window=126)

print(stats[["Annual return", "Sharpe ratio", "Max drawdown", "Beta"]])
print(drawdowns)
print(rolling.dropna().tail())
```

## Coverage

`mojopyfolio.timeseries` covers:

- `max_drawdown`, `annual_return`, `annual_volatility`, `calmar_ratio`,
  `omega_ratio`, `sortino_ratio`, `downside_risk`, `sharpe_ratio`,
  `stability_of_timeseries`, `tail_ratio`, `common_sense_ratio`, and
  `value_at_risk`
- `alpha_beta`, `alpha`, `beta`, `var_cov_var_normal`, `gross_lev`, and
  `perf_stats`
- `cum_returns`, `cum_returns_final`, `drawdown_series`, `aggregate_returns`,
  `normalize`, `rolling_beta`, `rolling_volatility`, and `rolling_sharpe`
- `get_max_drawdown_underwater`, `get_max_drawdown`, `get_top_drawdowns`, and
  `gen_drawdown_table`
- `calc_bootstrap`, `calc_distribution_stats`, `perf_stats_bootstrap`,
  `simulate_paths`, `summarize_paths`, and `forecast_cone_bootstrap`

`mojopyfolio.pos` covers percent allocation, top long/short/absolute positions,
position concentration, sector exposure, and long/short/net exposure.
`mojopyfolio.txn` covers transaction volume, slippage adjustment, and turnover.

Not covered are pyfolio's plotting and tear-sheet layers, capacity analysis,
round-trip trade extraction, performance attribution, online data loaders,
`rolling_regression`, interesting-period lookup, legacy transaction-object
normalization, and Zipline-specific position extraction. Those are mainly
visualization, I/O, or framework integration rather than compute-heavy
portfolio kernels.

## Upstream parity

The original Quantopian `pyfolio` release is 0.9.2 and is not compatible with
the repository's Python 3.13 environment. Tests therefore import the real
`pyfolio` module from `pyfolio-reloaded` 0.9.7, the maintained continuation
available on conda-forge. Its API descends directly from Quantopian pyfolio.

There are 45 pytest cases comparing scalar values, complete Series and
DataFrames, indices, column types, NaNs, drawdown dates, bootstrap samples,
positions, transactions, invalid FFI dimensions, SIMD tails, the parallel
threshold, and edge cases.
The conda-forge build currently resolves empyrical 0.5.9, which references
NumPy's removed `NINF` alias; the test harness supplies that alias only while
calling upstream. mojo-pyfolio itself does not depend on empyrical.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64. Each number is the best of three runs against `pyfolio-reloaded` 0.9.7
on identical float64 inputs.

| benchmark | mojo-pyfolio | pyfolio | speedup |
| --- | ---: | ---: | ---: |
| `max_drawdown`, 5M returns | 11.16 ms | 224.32 ms | 20.10x |
| `cum_returns`, 5M returns | 37.57 ms | 232.56 ms | 6.19x |
| `rolling_volatility`, 2M window 126 | 28.04 ms | 150.35 ms | 5.36x |
| `rolling_beta`, 3k window 126 | 0.96 ms | 1439.94 ms | 1496.81x |
| `summarize_paths`, 20k x 252 | 12.86 ms | 132.00 ms | 10.26x |
| `perf_stats`, 5M returns | 287.11 ms | 3194.18 ms | 11.13x |

The rolling-beta gap is unusually large because upstream iterates over pandas
label slices and runs a fresh regression for every row. The Mojo kernel keeps
five rolling sums and does constant work per row. `perf_stats` computes its
second through fourth central moments in two SIMD passes and parallelizes those
passes for large inputs; it also selects both tail percentiles in one
NumPy call. Maximum drawdown uses a dedicated recurrence instead of running the
full performance-summary kernel, and each rolling API allocates and computes
only the output it returns.

No GPU path is included. The array kernels are sequential wealth/rolling
recurrences or bandwidth-bound reductions. Host/device transfers and launch
overhead are not expected to help these kernels, so CPU remains the only
execution path.

## How it works

`src/kernels.mojo` is one compilation unit exporting a small C ABI. Python
passes array buffers as integer addresses through ctypes; each Mojo export
reconstructs an `UnsafePointer[Float64, AnyOrigin[mut=True]]`. Python-owned
NumPy arrays remain alive for each synchronous call, so there is no
cross-language allocator or cleanup protocol.

Inputs are C-contiguous float64 arrays. Existing compatible NumPy buffers cross
without a copy; returned pandas objects retain their Python-owned NumPy output
buffer without a defensive copy. Other array-like inputs are normalized at the
wrapper boundary. One-dimensional time series are streamed in chronological
order. Matrices use row-major layout, including bootstrap samples shaped
`(num_samples, num_days)`. Labeled Series and DataFrames are reconstructed in
Python with upstream-compatible indices and columns.

The wrapper validates dimensional compatibility and non-empty buffers before
constructing Mojo pointers. Sizes and addresses use pointer-width integers, and
no pointer is retained after a call returns.

The return-summary kernel fuses NaN-aware first and second moments, downside
and Omega accumulators, cumulative wealth, maximum drawdown, and cumulative-log
regression terms. Higher central moments use native-width float64 SIMD with a
scalar remainder loop and a bounded worker count for large arrays. Rolling
volatility, Sharpe, and beta use add-entering / subtract-leaving sums, reducing
pyfolio's rolling beta from repeated window-sized regressions to O(n).

MIT licensed.
