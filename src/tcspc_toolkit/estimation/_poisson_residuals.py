"""Signed Poisson residuals; undefined derivatives are availability, not zeros."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Residuals:
    values: np.ndarray
    derivatives: np.ndarray
    derivative_available: np.ndarray


def _poisson_residuals(observed, expected):
    """Return sign(y-mu)*sqrt(2*C_i) and dr/dmu, without expectation floors.

    At y=mu=0 the residual is zero but its expectation derivative is undefined.
    NaN in the internal derivative array denotes unavailable, never solver input.
    """
    y, mu = np.asarray(observed, float), np.asarray(expected, float)
    if (
        y.shape != mu.shape
        or y.ndim != 1
        or not np.all(np.isfinite(y) & np.isfinite(mu) & (y >= 0) & (mu >= 0))
    ):
        raise ValueError("finite nonnegative aligned vectors required")
    r = np.zeros_like(y)
    d = np.full_like(y, np.nan)
    zero = y == 0
    r[zero] = -np.sqrt(2.0) * np.sqrt(mu[zero])
    positive = (y > 0) & (mu > 0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        d[zero & (mu > 0)] = -1 / (np.sqrt(2.0) * np.sqrt(mu[zero & (mu > 0)]))
        delta = (mu[positive] - y[positive]) / y[positive]
        close = np.abs(delta) < 0.125
        # 2*(delta-log1p(delta))/delta**2; convergent series, including delta=0.
        z = delta[close]
        h = np.zeros_like(z)
        for power in range(22, -1, -1):
            h = h * z + 2 * (-1.0) ** power / (power + 2)
        rp = np.empty_like(delta)
        dp = np.empty_like(delta)
        root = np.sqrt(y[positive][close])
        rp[close] = -z * root * np.sqrt(h)
        dp[close] = -1 / (root * (1 + z) * np.sqrt(h))
        a, b = y[positive][~close], mu[positive][~close]
        # Scale before combining terms: intermediate a*log(a/b)+b may
        # overflow even when the centered objective is representable.
        c = np.empty_like(a)
        large = a >= b
        ratio = b[large] / a[large]
        c[large] = a[large] * (np.log(a[large]) - np.log(b[large]) + ratio - 1)
        ratio = a[~large] / b[~large]
        c[~large] = b[~large] * (
            1 + ratio * (np.log(a[~large]) - np.log(b[~large]) - 1)
        )
        rp[~close] = np.sign(a - b) * np.sqrt(2.0) * np.sqrt(c)
        dp[~close] = ((b - a) / rp[~close]) / b
        r[positive], d[positive] = rp, dp
    r[(y > 0) & (mu == 0)] = np.inf
    available = np.isfinite(r) & np.isfinite(d)
    for array in (r, d, available):
        array.setflags(write=False)
    return _Residuals(r, d, available)
