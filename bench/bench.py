"""Benchmarks against pyfolio-reloaded on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
np.NINF = -np.inf
np.PINF = np.inf
warnings.filterwarnings("ignore")

from mojopyfolio import timeseries as mojo  # noqa: E402
from mojopyfolio import __version__ as mojo_version  # noqa: E402
from pyfolio import __version__ as pyfolio_version  # noqa: E402
from pyfolio import timeseries as upstream  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(7)
    large = pd.Series(rng.normal(0.00003, 0.012, 5_000_000))
    rolling = pd.Series(rng.normal(0.00003, 0.012, 2_000_000))
    factor = pd.Series(
        rolling.to_numpy() * 0.3 + rng.normal(0, 0.008, len(rolling))
    )
    beta_returns = rolling.iloc[:3_000]
    beta_factor = factor.iloc[:3_000]
    samples = rng.normal(0.00003, 0.012, (20_000, 252))

    cases = [
        (
            "max_drawdown, 5M returns",
            lambda: mojo.max_drawdown(large),
            lambda: upstream.max_drawdown(large),
        ),
        (
            "cum_returns, 5M returns",
            lambda: mojo.cum_returns(large),
            lambda: upstream.cum_returns(large),
        ),
        (
            "rolling_volatility, 2M window 126",
            lambda: mojo.rolling_volatility(rolling, 126),
            lambda: upstream.rolling_volatility(rolling, 126),
        ),
        (
            "rolling_beta, 3k window 126",
            lambda: mojo.rolling_beta(beta_returns, beta_factor, 126),
            lambda: upstream.rolling_beta(beta_returns, beta_factor, 126),
        ),
        (
            "summarize_paths, 20k x 252",
            lambda: mojo.summarize_paths(samples),
            lambda: upstream.summarize_paths(samples),
        ),
        (
            "perf_stats, 5M returns",
            lambda: mojo.perf_stats(large),
            lambda: upstream.perf_stats(large),
        ),
    ]

    mojo.max_drawdown(large.iloc[:100])
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(
        f"Comparison: mojo-pyfolio {mojo_version} vs "
        f"pyfolio-reloaded {pyfolio_version}"
    )
    print("Timing: best of 3 runs on identical float64 inputs")
    print()
    print("| benchmark | mojo-pyfolio | pyfolio | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function in cases:
        mojo_seconds = timeit(mojo_function)
        upstream_seconds = timeit(upstream_function)
        speedup = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
