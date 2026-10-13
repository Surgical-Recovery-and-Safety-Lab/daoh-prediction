"""Generic non-parametric bootstrap utilities for confidence intervals."""

from collections.abc import Callable

import numpy as np

from src.constants import BOOTSTRAP_SEED, CI_LEVEL, N_BOOT


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], np.ndarray | float],
    n_samples: int,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Evaluate a statistic on bootstrap resamples of the sample indices.

    A new random generator is created from `seed` on every call, so every
    model evaluated with the same `n_samples` and `seed` is resampled with
    exactly the same indices.

    Parameters
    ----------
    statistic : callable
        Function taking an integer array of resampled indices of shape
        (n_samples,) and returning a float or an array of floats. It must
        index all the data it needs with these indices.
    n_samples : int
        Number of samples in the dataset.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Seed of the random generator.

    Returns
    -------
    np.ndarray of shape (n_boot, ...)
        Statistic value for each bootstrap iteration, stacked on axis 0.

    """
    rng = np.random.default_rng(seed)
    draws = [
        np.asarray(statistic(rng.integers(0, n_samples, size=n_samples)), dtype=float)
        for _ in range(n_boot)
    ]
    return np.stack(draws)


def percentile_interval(
    draws: np.ndarray, level: float = CI_LEVEL
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the percentile confidence interval of bootstrap draws.

    Parameters
    ----------
    draws : np.ndarray of shape (n_boot, ...)
        Bootstrap draws, as returned by `bootstrap_statistic`.
    level : float, default=0.95
        Confidence level of the interval.

    Returns
    -------
    lower : np.ndarray of shape (...)
        Lower bound of the interval.
    upper : np.ndarray of shape (...)
        Upper bound of the interval.

    """
    tail = 100 * (1.0 - level) / 2.0
    lower, upper = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
    return lower, upper
