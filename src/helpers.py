"""Helper functions for conference paper results: CRPS via the `scores`
package, region filtering, and LaTeX table generation.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper
from ordboost.metrics import (
    baseline_distribution,
    pit_diagnostics,
)
from scores.probability import PitFcstAtObs, crps_cdf
from src.bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOT,
    bootstrap_statistic,
    percentile_interval,
)
from src.helpers_ngboost import (
    NGBoostDistAdapter,
    cdf_grid_ngboost,
    crps_ngboost_samples,
    pit_diagnostics_ngboost,
)


def wrap_ngboost_pred_dist(ngb_pred_dist) -> NGBoostDistAdapter:
    """Wrap the output of `NGBRegressor.pred_dist(X)` in an adapter.

    Parameters
    ----------
    ngb_pred_dist
        NGBoost predicted distribution. Its `dist` attribute must be a scipy
        frozen distribution.

    Returns
    -------
    NGBoostDistAdapter
        Adapter exposing the ordboost distribution interface.

    """
    return NGBoostDistAdapter(ngb_pred_dist.dist)


def sample_crps(y_true, dist) -> np.ndarray:
    """Compute the per-sample CRPS of a predictive distribution.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        CRPS of each sample.

    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(dist, NGBoostDistAdapter):
        return crps_ngboost_samples(y_true_arr, dist)

    fcst = xr.DataArray(
        dist.grid_cdf,
        dims=["sample", "threshold"],
        coords={"threshold": dist.grid_y},
    )
    obs = xr.DataArray(y_true_arr, dims=["sample"])
    result = crps_cdf(fcst, obs, threshold_dim="threshold", preserve_dims=["sample"])
    return result.total.values


def compute_sample_cdfs(dist, indices, grid_y):
    """Extract the CDF at selected indices.

    Parameters
    ----------
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Distribution object.
    indices : dict[int, int]
        Dictionary with {index: daoh}.
    grid_y : array-like
        Grid points to evaluate the CDF on for NGBoost. Ignored for ordboost
        distributions, which use their own grid.

    Returns
    -------
    cdfs : dict[str, Any]
        Dictionary with:
        "grid_y": grid_y
        "cdfs": {daoh: cdf_array}

    """
    cdfs = {"grid_y": grid_y, "cdfs": {}}

    # Extract proper CDF grids
    if isinstance(dist, NGBoostDistAdapter):
        cdf_grid = cdf_grid_ngboost(dist, grid_y)
    else:
        cdf_grid = dist.grid_cdf

    for index, daoh in indices.items():  # Loop over indices
        cdfs["cdfs"][daoh] = cdf_grid[index]
    return cdfs


def save_table(latex_str, path: str | Path = "table.txt"):
    """Write a LaTeX table string to a text file.

    Parameters
    ----------
    latex_str : str
        LaTeX source of the table.
    path : str or Path, default="table.txt"
        Destination file. Overwritten if it exists.

    """
    with open(path, "w") as f:
        f.write(latex_str + "\n")


def pit_histogram(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    mapper: BaseBinMapper | None = None,
    n_bins: int = 20,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Compute the PIT histogram and alpha score for a model, with 95% CIs.

    The PIT intervals are computed once. As they are rounded, they take few
    distinct values, so each bootstrap resample is evaluated as a weighted
    PIT over the distinct intervals (exactly equal to the PIT of the
    resampled data, but much faster).

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution or NGBoostDistAdapter
        Predicted distributions.
    mapper : BaseBinMapper, optional
        Bin mapper of the OrdBoost model. Required unless `dist` is an
        NGBoostDistAdapter.
    n_bins : int, default=20
        Number of histogram bins.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Seed of the bootstrap random generator. The same seed gives the same
        resamples for every model.

    Returns
    -------
    dict
        Dictionary with keys "alpha" (the PIT alpha score), "hist_values"
        (the histogram values), "alpha_ci" (lower and upper bound of the
        alpha score as a tuple) and "hist_ci" (lower and upper bounds of the
        histogram values as a tuple of arrays).

    Raises
    ------
    ValueError
        If `dist` is not an NGBoostDistAdapter and `mapper` is None.

    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(dist, NGBoostDistAdapter):
        pit = pit_diagnostics_ngboost(y_true_arr, dist)

    elif mapper is not None:
        pit = pit_diagnostics(y_true_arr, dist, mapper)

    else:
        raise ValueError("mapper needs to be provided with OrdBoost model.")

    alpha = pit.alpha_score()
    hist_values = pit.hist_values(n_bins)

    # Group the samples by distinct PIT interval
    endpoints = pit.pit_uniform_endpoints.values
    pairs, inverse = np.unique(endpoints.T, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    lower_da = xr.DataArray(pairs[:, 0], dims=["sample"])
    upper_da = xr.DataArray(pairs[:, 1], dims=["sample"])

    def statistic(idx):
        counts = np.bincount(inverse[idx], minlength=len(pairs)).astype(float)
        boot_pit = PitFcstAtObs(
            upper_da,
            fcst_at_obs_left=lower_da,
            weights=xr.DataArray(counts, dims=["sample"]),
        )
        return np.append(boot_pit.hist_values(n_bins).values, boot_pit.alpha_score())

    draws = bootstrap_statistic(statistic, len(y_true_arr), n_boot, seed)
    lower, upper = percentile_interval(draws)

    return {
        "alpha": alpha,
        "hist_values": hist_values,
        "alpha_ci": (float(lower[-1]), float(upper[-1])),
        "hist_ci": (lower[:-1], upper[:-1]),
    }


def marginal_calibration(
    y_true,
    dist,
    grid_y,
    mapper,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
):
    """Compute the marginal calibration curve for a model, with a 95% CI.

    The curve is the difference between the empirical CDF and the mean
    predicted CDF at each grid point. The per-sample contributions are
    computed once, and the confidence band is the pointwise percentile
    interval over bootstrap resamples of the test samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.
    grid_y : array-like
        Grid to evaluate the curve on. Only used for NGBoost.
    mapper : BaseBinMapper
        Bin mapper of the OrdBoost model. Only used for ordboost
        distributions, to handle the floor and ceiling atoms.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Seed of the bootstrap random generator. The same seed gives the same
        resamples for every model.

    Returns
    -------
    grid_y : np.ndarray of shape (n_points,)
        Grid points at which the curve is evaluated.
    calibration : np.ndarray of shape (n_points,)
        Empirical CDF minus mean predicted CDF.
    lower : np.ndarray of shape (n_points,)
        Lower bound of the 95% CI.
    upper : np.ndarray of shape (n_points,)
        Upper bound of the 95% CI.

    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(dist, NGBoostDistAdapter):
        grid = np.asarray(grid_y, dtype=float)
        cdf = cdf_grid_ngboost(dist, grid)
        indicator = (y_true_arr[:, None] <= grid[None, :]).astype(float)
    else:
        grid = np.asarray(dist.grid_y, dtype=float)
        cdf = np.array(dist.grid_cdf, dtype=float)
        indicator = (y_true_arr[:, None] < grid[None, :]).astype(float)
        if getattr(mapper, "floor_atom", False):
            cdf[:, 1] = 0.0  # left-hand limit at the floor atom
        if getattr(mapper, "ceiling_atom", False):
            indicator[:, -1] = 1.0  # by convention

    contribution = indicator - cdf  # Per-sample eCDF - CDF
    n_samples = len(y_true_arr)

    def statistic(idx):
        counts = np.bincount(idx, minlength=n_samples)
        return counts @ contribution / n_samples

    draws = bootstrap_statistic(statistic, n_samples, n_boot, seed)
    lower, upper = percentile_interval(draws)
    return grid, contribution.mean(axis=0), lower, upper


def coverage_sharpness_curve(
    y_true,
    dist,
    coverage_levels,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
):
    """Compute coverage, sharpness and Winkler score across coverage levels.

    The alpha convention matches ordboost.metrics: alpha = 1 - coverage / 100,
    e.g. coverage=90 gives alpha=0.10. The per-sample values are computed
    once for every level, and the 95% CIs are percentile intervals over
    bootstrap resamples of the test samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predicted distributions.
    coverage_levels : array-like
        Nominal central-interval coverage levels in percent (e.g. 90 for a
        90% interval).
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Seed of the bootstrap random generator. The same seed gives the same
        resamples for every model.

    Returns
    -------
    dict
        Dictionary with the keys "coverage_levels", "empirical_coverage",
        "winkler" and "sharpness" (mean interval width), each with one value
        per coverage level. Each metric also has a "<metric>_ci" key holding
        the (lower, upper) bounds of its 95% CI.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    coverage_levels = np.asarray(coverage_levels, dtype=float)
    alphas = 1.0 - coverage_levels / 100.0
    n_samples, n_levels = len(y_true_arr), len(alphas)

    covered = np.empty((n_samples, n_levels))
    width = np.empty((n_samples, n_levels))
    winkler = np.empty((n_samples, n_levels))

    for i, alpha in enumerate(alphas):
        lower, upper = dist.interval(alpha=alpha)
        covered[:, i] = (y_true_arr >= np.round(lower)) & (
            y_true_arr <= np.round(upper)
        )
        width[:, i] = upper - lower
        winkler[:, i] = (
            width[:, i]
            + (2.0 / alpha) * (lower - y_true_arr) * (y_true_arr < lower)
            + (2.0 / alpha) * (y_true_arr - upper) * (y_true_arr > upper)
        )

    per_sample = np.hstack([covered, width, winkler])

    def statistic(idx):
        counts = np.bincount(idx, minlength=n_samples)
        return counts @ per_sample / n_samples

    estimates = per_sample.mean(axis=0)
    draws = bootstrap_statistic(statistic, n_samples, n_boot, seed)
    lower, upper = percentile_interval(draws)

    results = {"coverage_levels": coverage_levels}
    names = ["empirical_coverage", "sharpness", "winkler"]
    for k, name in enumerate(names):
        block = slice(k * n_levels, (k + 1) * n_levels)
        results[name] = estimates[block]
        results[f"{name}_ci"] = (lower[block], upper[block])
    return results


def interval_coverage(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    alpha: float = 0.10,
) -> float:
    """Compute empirical coverage rate for a central prediction interval.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution
        Predicted continuous distributions.
    alpha : float, default=0.10
        Tail significance level (e.g., alpha=0.10 specifies a 90% interval).

    Returns
    -------
    float
        Proportion of true observations lying within predicted interval bounds.

    Raises
    ------
    ValueError
        If alpha not within (0.0, 1.0);

    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("Significance level 'alpha' must lie within (0.0, 1.0).")

    y_true_arr = np.asarray(y_true, dtype=float)
    lower, upper = dist.interval(alpha=alpha)

    covered = (
        (y_true_arr >= np.round(lower)) & (y_true_arr <= np.round(upper))
    ).astype(float)

    return float(np.mean(covered))


def estimate_patients(dist, ind: dict[int, int], alpha: float) -> dict[int, tuple]:
    """Estimate the DAOH and prediction interval of selected patients.

    The point prediction is the rounded median of the distribution.

    Parameters
    ----------
    dist : NGBoostDistAdapter or ContinuousPredictiveDistribution
        Predictive distribution.
    ind : dict[int, int]
        Dictionary of indices to check with associated true DAOH.
    alpha : float
        Tail significance level of the prediction interval (e.g. 0.05 for a
        95% interval).

    Returns
    -------
    dict[int, tuple]
        Dictionary with the index as key and a tuple (true DAOH, prediction,
        lower bound, upper bound) as value.

    """
    predictions = dist.median()
    results = {}
    l_bound, u_bound = dist.interval(alpha)

    for index, daoh in ind.items():
        results[index] = (
            daoh,
            np.round(predictions[index]),
            np.round(l_bound[index]),
            np.round(u_bound[index]),
        )

    return results


def _format_continuous(series):
    """Format a series as 'median (Q1--Q3)', ignoring NaNs.

    Parameters
    ----------
    series : pd.Series
        Continuous values to summarise.

    Returns
    -------
    str
        The formatted summary, or "--" if there are no valid values.

    """
    valid = series.dropna()
    if len(valid) == 0:
        return "--"
    median = np.median(valid)
    q1, q3 = np.percentile(valid, [25, 75])
    return f"{median:.1f} ({q1:.1f}--{q3:.1f})"


def _format_n_pct(n, total):
    """Format a count as 'n (pct%)' with a LaTeX-escaped percent sign.

    Parameters
    ----------
    n : int
        Count of interest.
    total : int
        Full size of the dataset (missing values included).

    Returns
    -------
    str
        The formatted count and percentage. The percentage is 0 if `total`
        is not positive.

    """
    pct = 100 * n / total if total > 0 else 0.0
    return f"{n:,} ({pct:.1f}\\%)"


def _categorical_rows(datasets, col, categories=None):
    """Build the table rows for a categorical column across datasets.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        Datasets to summarise.
    col : str
        Categorical column name.
    categories : list, optional
        Category order. Defaults to the sorted union of the categories found
        in the datasets.

    Returns
    -------
    list[tuple[str, list[str]]]
        One (label, cells) row per category, plus a trailing "Missing" row
        if any dataset has NaNs in this column.

    """
    if categories is None:
        categories = sorted(
            set().union(*[set(d[col].dropna().unique()) for d in datasets])
        )

    rows = []
    for cat in categories:
        counts = [int((d[col] == cat).sum()) for d in datasets]
        totals = [len(d) for d in datasets]
        rows.append((str(cat), [_format_n_pct(n, t) for n, t in zip(counts, totals)]))

    missing = [int(d[col].isna().sum()) for d in datasets]
    if any(m > 0 for m in missing):
        totals = [len(d) for d in datasets]
        rows.append(("Missing", [_format_n_pct(n, t) for n, t in zip(missing, totals)]))

    return rows


def generate_patient_characteristics_table(
    X_train,
    X_test,
    continuous_features,
    categorical_features,
    feature_labels=None,
    category_orders=None,
    caption="Patient characteristics by dataset.",
    label="tab:patient_chars",
):
    """Build a LaTeX longtable/booktabs patient characteristics table.

    The table compares the train and test splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with the original (pre-OrdinalEncoder) categorical
        values, so category labels render as text rather than integer codes.
    X_test : pd.DataFrame
        Test features, with the same format as `X_train`.
    continuous_features : list[str]
        Columns summarised as median (IQR). Only the first one is used, and
        it is reported as the age.
    categorical_features : list[str]
        Columns summarised as one row per category, N (%).
    feature_labels : dict[str, str], optional
        Column name -> display name. Defaults to the column name.
    category_orders : dict[str, list], optional
        Column name -> explicit category order, e.g. {"trauma": ["No","Yes"]},
        so binary features don't sort alphabetically as ["No","Yes"] vs
        ["Yes","No"] depending on the data.
    caption : str, default="Patient characteristics by dataset."
        Table caption.
    label : str, default="tab:patient_chars"
        LaTeX label of the table.

    Returns
    -------
    str
        Complete LaTeX longtable source.
    """
    feature_labels = feature_labels or {}
    category_orders = category_orders or {}
    datasets = [X_train, X_test]
    n_train, n_test = (len(d) for d in datasets)

    header = (
        r"\textbf{Features} & "
        rf"\shortstack[l]{{\textbf{{Train (2010--2022)}} \\ N = {n_train:,}}} & "
        rf"\shortstack[l]{{\textbf{{Test (2023--2024)}} \\ N = {n_test:,}}} \\"
    )
    lines = [
        r"\begin{longtable}{l l l}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule",
        header,
        r"\midrule\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{l}{\textit{(continued)}} \\",
        r"\toprule",
        header,
        r"\midrule\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    # Age first, as specified
    age_col = continuous_features[0]
    age_label = feature_labels.get(age_col, "Age")
    cells = [_format_continuous(d[age_col]) for d in datasets]
    lines.append(rf"{age_label}, median (IQR) & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")

    for col in categorical_features:
        label_txt = feature_labels.get(col, col)
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{label_txt}}}}} \\")
        for cat_label, cells in _categorical_rows(
            datasets, col, category_orders.get(col)
        ):
            lines.append(rf"\quad {cat_label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)
