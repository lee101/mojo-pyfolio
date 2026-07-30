"""ctypes loader for the Mojo portfolio kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYFOLIO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyfolio.so"
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpf_cum_returns": ([I, I, I, I, F], None),
    "mpf_max_drawdown": ([I, I, I], None),
    "mpf_drawdown_series": ([I, I, I], None),
    "mpf_return_summary": ([I, I, F, F, I], None),
    "mpf_central_moments": ([I, I, I], None),
    "mpf_factor_summary": ([I, I, I, I], None),
    "mpf_rolling_stats": ([I, I, I, I, I, F, I], None),
    "mpf_rolling_beta": ([I, I, I, I, I], None),
    "mpf_summarize_paths": ([I, I, I, I, I, F], None),
}

_handle: ctypes.CDLL | None = None


def _build() -> None:
    script = os.path.join(ROOT, "build", "build.sh")
    process = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if process.returncode:
        raise RuntimeError((process.stderr or process.stdout).strip())


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            _build()
        _handle = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_handle, name)
            function.argtypes = argtypes
            function.restype = restype
    return _handle


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def address(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if values.dtype != np.float64 or not values.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    if values.size == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    pointer = int(values.ctypes.data)
    if pointer == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return pointer


def summary(values, required_return: float = 0.0, omega_offset: float = 0.0):
    source = f64(values).ravel()
    result = np.empty(11, dtype=np.float64)
    lib().mpf_return_summary(
        address(source), source.size, required_return, omega_offset, address(result)
    )
    return result


def max_drawdown_value(values) -> float:
    source = f64(values).ravel()
    result = np.empty(1, dtype=np.float64)
    lib().mpf_max_drawdown(address(source), source.size, address(result))
    return float(result[0])


def central_moments(values):
    source = f64(values).ravel()
    if source.size == 0:
        raise ValueError("central moments require at least one value")
    result = np.empty(24, dtype=np.float64)
    lib().mpf_central_moments(address(source), source.size, address(result))
    return result[:4]


def factor_summary(returns, factor_returns):
    left = f64(returns).ravel()
    right = f64(factor_returns).ravel()
    if left.size != right.size:
        raise ValueError("returns and factor_returns must have equal size")
    if left.size == 0:
        return np.zeros(5, dtype=np.float64)
    result = np.empty(5, dtype=np.float64)
    lib().mpf_factor_summary(address(left), address(right), left.size, address(result))
    return result
