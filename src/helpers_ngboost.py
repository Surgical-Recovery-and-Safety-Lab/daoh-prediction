"""Helper functions adapting ordboost's evaluation pipeline to NGBoost's
parametric (scipy-based) predictive distributions, so plotting.py can be
reused completely unchanged.
"""

from __future__ import annotations

import numpy as np
import properscoring as ps
import xarray as xr
from scores.probability import PitFcstAtObs


class NGBoostDistAdapter:
    """Duck-typed wrapper exposing ordboost.PredictiveDistribution-style
    methods over an NGBoost predicted distribution's underlying scipy
    frozen distribution.

    Parameters
    ----------
    scipy_frozen_dist : scipy.stats rv_frozen
        The `.dist` attribute of an NGBoost `pred_dist(X)` result -- a
        scipy frozen distribution with array-valued parameters, one per
        sample.
    """

    def __init__(self, scipy_frozen_dist):
        self._dist = scipy_frozen_dist

    def mean(self) -> np.ndarray:
        """Compute the per-sample mean.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Mean of each predicted distribution.

        """
        return np.clip(self._dist.mean(), a_min=0, a_max=89)

    def median(self) -> np.ndarray:
        """Compute the per-sample median.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Median of each predicted distribution.

        """
        return np.clip(self._dist.median(), a_min=0, a_max=89)

    def ppf(self, q) -> np.ndarray:
        """Compute the per-sample quantile at level(s) q.

        Parameters
        ----------
        q : float or array-like
            Quantile level(s) in [0, 1].

        Returns
        -------
        np.ndarray
            Quantile of each predicted distribution at `q`.

        """
        return np.asarray(self._dist.ppf(q))

    def interval(self, alpha: float = 0.10):
        """Compute the central prediction interval at significance level alpha.

        Scipy's own `.interval()` takes a *confidence* level (the fraction
        covered), the inverse convention from ordboost's alpha (tail
        significance). It is converted here so callers use the same alpha
        convention as ordboost.PredictiveDistribution throughout.

        Parameters
        ----------
        alpha : float, default=0.10
            Tail significance level (e.g. 0.10 specifies a 90% interval).

        Returns
        -------
        lower : np.ndarray of shape (n_samples,)
            Lower bound of the interval for each sample.
        upper : np.ndarray of shape (n_samples,)
            Upper bound of the interval for each sample.

        """
        lower, upper = self._dist.interval(1.0 - alpha)
        return np.clip(lower, a_min=0, a_max=89), np.clip(upper, a_min=0, a_max=89)

    def cdf(self, y) -> np.ndarray:
        """Evaluate the CDF at y.

        Parameters
        ----------
        y : float or array-like of shape (n_samples,)
            Point(s) to evaluate at. A scalar broadcasts across all samples;
            an array evaluates each sample at its own y.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            CDF values.

        """
        return np.clip(self._dist.cdf(y), a_min=0, a_max=89)

    def __getitem__(self, mask) -> NGBoostDistAdapter:
        """Slice the samples with a boolean mask.

        The underlying scipy distribution is re-frozen on the masked
        parameter arrays.

        Parameters
        ----------
        mask : array-like of bool of shape (n_samples,)
            Mask selecting the samples to keep.

        Returns
        -------
        NGBoostDistAdapter
            Adapter over the selected samples.

        """
        mask = np.asarray(mask, dtype=bool)
        new_args = tuple(
            np.asarray(a)[mask] if np.ndim(a) > 0 else a for a in self._dist.args
        )
        new_kwds = {
            k: (np.asarray(v)[mask] if np.ndim(v) > 0 else v)
            for k, v in self._dist.kwds.items()
        }
        return NGBoostDistAdapter(self._dist.dist(*new_args, **new_kwds))


def crps_ngboost_per_sample(y_true, ngb_dist) -> float:
    """Mean CRPS of an NGBoostDistAdapter (lower is better)."""
    dist = ngb_dist.dist
    mean = np.clip(dist.mean(), a_min=0, a_max=89)
    sig = dist.std()
    scores = np.asarray(ps.crps_gaussian(y_true, mu=mean, sig=sig))
    return np.mean(scores)


def crps_ngboost_score(y_true, ngb_dist) -> float:
    """Mean CRPS of an NGBoostDistAdapter (lower is better)."""
    return float(np.mean(crps_ngboost_per_sample(y_true, ngb_dist)))


def crps_ngboost_samples(y_true, dist: NGBoostDistAdapter) -> np.ndarray:
    """Compute the per-sample CRPS of an NGBoostDistAdapter.

    The CRPS is closed-form for a Normal distribution (fast), otherwise it
    uses per-sample numerical quadrature via properscoring (slower, consider
    subsampling large test sets).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter
        Predicted distributions.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        CRPS of each sample.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    frozen = dist._dist
    if getattr(frozen.dist, "name", "") == "norm":
        return np.asarray(
            ps.crps_gaussian(y_true_arr, mu=frozen.mean(), sig=frozen.std())
        )

    scores = np.empty(len(y_true_arr))
    for i, y in enumerate(y_true_arr):
        row = dist[np.arange(len(y_true_arr)) == i]
        scores[i] = ps.crps_quadrature(
            y, row._dist.cdf, xmin=row.ppf(1e-6)[0], xmax=row.ppf(1 - 1e-6)[0]
        )
    return scores


def crps_ngboost(y_true, dist: NGBoostDistAdapter) -> float:
    """Compute the mean CRPS of an NGBoostDistAdapter.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter
        Predicted distributions.

    Returns
    -------
    float
        CRPS averaged over the samples.

    """
    return float(np.mean(crps_ngboost_samples(y_true, dist)))


def marginal_calibration_curve_ngboost(y_true, dist, grid_y):
    """Compute the marginal calibration curve of an NGBoostDistAdapter.

    The curve is evaluated over a caller-supplied grid, as no fitted grid is
    available.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter
        Predicted distributions.
    grid_y : array-like
        Grid points to evaluate the curve on.

    Returns
    -------
    grid_y : np.ndarray
        The evaluation grid.
    diff : np.ndarray
        Empirical CDF minus mean predicted CDF at each grid point.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    mean_cdf = np.array([np.mean(dist.cdf(y)) for y in grid_y])
    empirical_cdf = np.array([np.mean(y_true_arr <= y) for y in grid_y])
    return grid_y, empirical_cdf - mean_cdf


def pit_diagnostics_ngboost(y_true, dist: NGBoostDistAdapter) -> PitFcstAtObs:
    """Compute exact PIT diagnostics for an NGBoostDistAdapter.

    This uses the same `scores.probability.PitFcstAtObs` machinery as
    `ordboost.metrics.pit_diagnostics`. Values are rounded to 2 decimal
    places before construction, matching the workaround used on the OrdBoost
    side to limit the memory use of PitFcstAtObs.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter
        Predicted distributions.

    Returns
    -------
    PitFcstAtObs
        PIT diagnostics object.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    fcst_at_obs = np.round(dist.cdf(y_true_arr), 2)

    fcst_da = xr.DataArray(fcst_at_obs, dims=["sample"])

    return PitFcstAtObs(fcst_da)


def cdf_grid_ngboost(dist: NGBoostDistAdapter, grid_y) -> np.ndarray:
    """Evaluate an NGBoostDistAdapter's CDF on a shared grid.

    The result is directly analogous to
    `ContinuousPredictiveDistribution.grid_cdf`.

    Parameters
    ----------
    dist : NGBoostDistAdapter
        Predicted distributions.
    grid_y : array-like
        Same grid used for the ordboost distribution, e.g.
        ordboost_dist.grid_y, so both models' CDFs can be compared or
        plotted point-for-point on identical x-values.

    Returns
    -------
    np.ndarray of shape (n_samples, len(grid_y))
        CDF of every sample at every grid point.

    """
    grid_y = np.asarray(grid_y, dtype=float)
    n_samples = len(dist.mean())
    grid_cdf = np.empty((n_samples, len(grid_y)), dtype=float)
    for j, y in enumerate(grid_y):
        grid_cdf[:, j] = dist.cdf(y)
    return grid_cdf
