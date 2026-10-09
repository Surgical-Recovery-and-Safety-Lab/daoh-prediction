"""Plot functions for conference paper figures."""

import matplotlib.pyplot as plt
import numpy as np
from medpipe.visualisation.themes import MedpipeTheme

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    }
)
THEME = MedpipeTheme()
COLOURS = {
    "ngboost": THEME.palette[0],
    "ordboost": THEME.palette[1],
    "4-bins": THEME.palette[0],
    "8-bins": THEME.palette[1],
    "12-bins": THEME.palette[2],
    "16-bins": THEME.palette[3],
    "20-bins": THEME.palette[4],
    "uniform": THEME.palette[0],
    "quantile": THEME.palette[1],
    "custom": THEME.palette[2],
    "median": THEME.palette[2],
    "mean": THEME.palette[3],
    "continuous": THEME.palette[4],
}


def plot_binned_vs_continuous(
    y_cont,
    y_binned,
    bin_edges,
    colors,
    n_fine_bins: int = 90,
    density: bool = False,
    log_scale: bool = True,
    save_path=None,
):
    """Plot the continuous target and its binned version.

    Both distributions share the DAOH-value x-axis, so they are directly
    comparable.

    Parameters
    ----------
    y_cont : array-like
        Continuous target values (e.g. DAOH-90 in days).
    y_binned : array-like
        Integer bin labels for the same samples, 0..n_bins-1.
    bin_edges : array-like
        Edges used to create y_binned. Length must be n_bins + 1.
    colors : list[str]
        Colors to plot with.
    n_fine_bins : int, default=90
        Number of bins for the continuous histogram.
    density : bool, default=False
        If True, plot binned bars as counts / bin_width, so bar *area*
        (not just height) is comparable across very uneven bin widths. If
        False, plot raw counts.
    log_scale : bool, default=True
        Use a log y-axis.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : np.ndarray of matplotlib.axes.Axes
        The two stacked axes (continuous on top, binned below).

    """
    y_cont = np.asarray(y_cont, dtype=float)
    y_binned = np.asarray(y_binned, dtype=int)
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)

    # Continuous distribution: fine fixed-width histogram.
    axes[0].hist(
        y_cont,
        bins=n_fine_bins,
        color=colors[0],
        edgecolor="black",
        label=r"Continuous $DAOH_{90}$",
    )

    # Binned distribution: bars positioned at the real bin_edges values,
    # each spanning its true [edges[k], edges[k+1]) width.
    counts = np.bincount(y_binned, minlength=n_bins).astype(float)
    widths = np.diff(edges)
    heights = counts / widths if density else counts

    axes[1].bar(
        edges[:-1],
        heights[1:],
        width=widths,
        align="edge",
        color=colors[1],
        edgecolor="black",
        label=r"Binned $DAOH_{90}$",
    )

    if log_scale:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    axes[1].set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    axes[0].set_ylabel("Count / bin width" if density else "Count", fontweight="bold")
    axes[1].set_ylabel("Count / bin width" if density else "Count", fontweight="bold")
    axes[0].set_title(
        r"Continuous and binned $\mathbf{DAOH_{90}}$ distributions", fontweight="bold"
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    axes[0].set_ylim(bottom=10.5)
    axes[1].set_ylim(bottom=10.5)

    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_sample_cdfs(model_results, colors, title, save_path=None):
    """Plot example CDFs for different DAOH values.

    Parameters
    ----------
    model_results : dict
        Output of `compute_sample_cdfs`, with keys "grid_y" and "cdfs"
        ({daoh: cdf_array}).
    colors : list[str]
        One colour per CDF.
    title : str
        Model name used in the plot title.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    """

    fig, ax = plt.subplots(figsize=(7, 5), sharex=True)
    grid_y = model_results["grid_y"]

    for i, (daoh, cdf) in enumerate(model_results["cdfs"].items()):
        ax.plot(
            grid_y,
            cdf,
            label=r"$DAOH_{90}$ = " + f"{daoh}",
            color=colors[i],
            linewidth=2,
        )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Median",
    )

    ax.set_ylim(top=1.4)
    ax.set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.set_title(f"{title} cumulative distribution functions", fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_pit_histogram_grouped(
    model_hists,
    display_labels=None,
    n_bins=20,
    save_path=None,
    ax=None,
):
    """Plot randomized PIT histograms as grouped bars on a shared axis.

    Each PIT bin holds one bar per model. Sized for single-column placement.

    Parameters
    ----------
    model_hists : dict
        Model name -> output of `pit_histogram` (keys "alpha" and
        "hist_values"). The 95% CIs ("alpha_ci" and "hist_ci"), if present,
        are shown as error bars on the bars and in the legend.
    display_labels : dict[str, str], optional
        Model name -> display name. Defaults to the model name.
    n_bins : int, default=20
        Number of PIT bins, must match the histograms.
    save_path : str or Path, optional
        If given, the figure is saved there.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    """
    display_labels = display_labels or {}
    models = list(model_hists.keys())
    n_groups = len(models)

    style = {
        "ngboost": COLOURS["ngboost"],
        "ordboost": COLOURS["ordboost"],
        "uniform": COLOURS["uniform"],
        "quantile": COLOURS["quantile"],
        "custom": COLOURS["custom"],
        "continuous": COLOURS["continuous"],
        "median": COLOURS["median"],
        "mean": COLOURS["mean"],
        "4-bins": COLOURS["4-bins"],
        "8-bins": COLOURS["8-bins"],
        "12-bins": COLOURS["12-bins"],
        "16-bins": COLOURS["16-bins"],
        "20-bins": COLOURS["20-bins"],
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    bin_width = 1.0 / n_bins
    group_width = bin_width * 0.9
    bar_width = group_width / n_groups

    for i, model in enumerate(models):
        hist = model_hists[model]["hist_values"]
        bin_centres = hist["bin_centre"].values
        bin_lefts = bin_centres - bin_width / 2.0
        bar_starts = bin_lefts + i * bar_width + (bin_width - group_width) / 2.0

        alpha = float(model_hists[model]["alpha"].values)
        label = (
            f"{display_labels.get(model, model)}"
            + r" $\alpha$ score: "
            + f"{alpha:.3f}"
        )
        if "alpha_ci" in model_hists[model]:
            a_low, a_high = model_hists[model]["alpha_ci"]
            label += f" [{a_low:.3f}, {a_high:.3f}]"
        ax.bar(
            bar_starts,
            hist.values,
            width=bar_width,
            align="edge",
            color=style.get(model, None),
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )

        if "hist_ci" in model_hists[model]:
            lower, upper = model_hists[model]["hist_ci"]
            centres = bar_starts + bar_width / 2.0
            yerr = np.vstack(
                [
                    np.clip(hist.values - lower, 0, None),
                    np.clip(upper - hist.values, 0, None),
                ]
            )
            ax.errorbar(
                centres,
                hist.values,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=1.5,
            )

    ax.axhline(
        1.0 / n_bins,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Uniform",
    )

    ax.set_xlabel("PIT value", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_title("PIT histograms", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Below the axes, the CIs make the labels too long to sit over the bars
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_marginal_calibration(model_results, display_labels, save_path=None, ax=None):
    """Plot the marginal calibration curves.

    Parameters
    ----------
    model_results : dict
        Model name -> (grid_y, diff, lower, upper), as returned by
        `marginal_calibration`. The lower and upper bounds of the 95% CI are
        drawn as a shaded area if present.
    display_labels : dict[str, str]
        Model name -> display name.
    save_path : str or Path, optional
        If given, the figure is saved there.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    style = {
        "ngboost": {"color": COLOURS["ngboost"]},
        "ordboost": {"color": COLOURS["ordboost"]},
        "uniform": {"color": COLOURS["uniform"]},
        "quantile": {"color": COLOURS["quantile"]},
        "custom": {"color": COLOURS["custom"]},
        "continuous": {"color": COLOURS["continuous"]},
        "median": {"color": COLOURS["median"]},
        "mean": {"color": COLOURS["mean"]},
        "4-bins": {"color": COLOURS["4-bins"]},
        "8-bins": {"color": COLOURS["8-bins"]},
        "12-bins": {"color": COLOURS["12-bins"]},
        "16-bins": {"color": COLOURS["16-bins"]},
        "20-bins": {"color": COLOURS["20-bins"]},
    }

    for label, results in model_results.items():
        (line,) = ax.plot(
            results[0],
            results[1],
            linewidth=2,
            label=display_labels.get(label, label),
            **style.get(label, {}),
        )
        if len(results) >= 4:
            ax.fill_between(
                results[0],
                results[2],
                results[3],
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$\mathbf{DAOH_{90}}$ (days)", fontweight="bold")
    ax.set_ylabel(r"eCDF - $\mathbf{\overline{CDF}}$", fontweight="bold")
    ax.set_title("Marginal calibration curves", fontweight="bold")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def _plot_coverage_metric(
    model_results,
    key,
    ylabel,
    title,
    display_labels=None,
    save_path=None,
    scale=1.0,
    ylim=None,
    ideal_line=False,
):
    """Plot one metric against the nominal coverage level, with CI error bars.

    One line is drawn per model. The models are shifted slightly along the x
    axis so that overlapping error bars remain readable.

    Parameters
    ----------
    model_results : dict
        Model name -> output of `coverage_sharpness_curve`.
    key : str
        Metric to plot. Its 95% CI is read from the "<key>_ci" entry, and
        error bars are skipped if it is missing.
    ylabel : str
        Label of the y axis.
    title : str
        Title of the plot.
    display_labels : dict[str, str], optional
        Model name -> display name. Defaults to the model name.
    save_path : str or Path, optional
        If given, the figure is saved there.
    scale : float, default=1.0
        Factor applied to the metric and its CI, e.g. 100 for percentages.
    ylim : tuple[float, float], optional
        Limits of the y axis.
    ideal_line : bool, default=False
        Draw the y = x diagonal, used for the coverage reliability plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    """
    display_labels = display_labels or {}
    style = {
        "ngboost": {"color": COLOURS["ngboost"], "marker": "o"},
        "ordboost": {"color": COLOURS["ordboost"], "marker": "s"},
        "uniform": {"color": COLOURS["uniform"], "marker": "s"},
        "quantile": {"color": COLOURS["quantile"], "marker": "s"},
        "custom": {"color": COLOURS["custom"], "marker": "s"},
        "continuous": {"color": COLOURS["continuous"], "marker": "s"},
        "mean": {"color": COLOURS["mean"], "marker": "s"},
        "median": {"color": COLOURS["median"], "marker": "s"},
        "4-bins": {"color": COLOURS["4-bins"], "marker": "s"},
        "8-bins": {"color": COLOURS["8-bins"], "marker": "s"},
        "12-bins": {"color": COLOURS["12-bins"], "marker": "s"},
        "16-bins": {"color": COLOURS["16-bins"], "marker": "s"},
        "20-bins": {"color": COLOURS["20-bins"], "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    n_models = len(model_results)
    for i, (model, results) in enumerate(model_results.items()):
        x = np.asarray(results["coverage_levels"], dtype=float)
        x = x + 0.4 * (i - (n_models - 1) / 2.0)  # Dodge overlapping models
        y = np.asarray(results[key]) * scale

        yerr = None
        if f"{key}_ci" in results:
            lower, upper = (np.asarray(b) * scale for b in results[f"{key}_ci"])
            yerr = np.vstack([np.clip(y - lower, 0, None), np.clip(upper - y, 0, None)])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=display_labels.get(model, model),
            linewidth=2,
            elinewidth=1,
            capsize=2,
            **style.get(model, {}),
        )

    if ideal_line:
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Ideal",
        )

    ax.set_xlabel("Nominal coverage (%)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")

    ax.set_xlim(0, 100)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_coverage(model_results, display_labels=None, save_path=None):
    """Plot the empirical coverage against the nominal coverage (figure 5).

    One line with 95% CI error bars is drawn per model, along with the ideal
    diagonal.

    Parameters
    ----------
    model_results : dict
        Model name -> output of `coverage_sharpness_curve`.
    display_labels : dict[str, str], optional
        Model name -> display name. Defaults to the model name.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    """
    return _plot_coverage_metric(
        model_results,
        "empirical_coverage",
        "Empirical coverage (%)",
        "Coverage reliability",
        display_labels=display_labels,
        save_path=save_path,
        scale=100.0,
        ylim=(0, 100),
        ideal_line=True,
    )


def plot_sharpness(model_results, display_labels=None, save_path=None):
    """Plot the sharpness across nominal coverage levels (figure 4).

    One line with 95% CI error bars is drawn per model.

    Parameters
    ----------
    model_results : dict
        Model name -> output of `coverage_sharpness_curve`.
    display_labels : dict[str, str], optional
        Model name -> display name. Defaults to the model name.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    """
    return _plot_coverage_metric(
        model_results,
        "sharpness",
        "Mean interval width (days)",
        "Sharpness",
        display_labels=display_labels,
        save_path=save_path,
    )


def plot_winkler(model_results, display_labels=None, save_path=None):
    """Plot the Winkler score across nominal coverage levels.

    One line with 95% CI error bars is drawn per model.

    Parameters
    ----------
    model_results : dict
        Model name -> output of `coverage_sharpness_curve`.
    display_labels : dict[str, str], optional
        Model name -> display name. Defaults to the model name.
    save_path : str or Path, optional
        If given, the figure is saved there.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    """
    return _plot_coverage_metric(
        model_results,
        "winkler",
        "Winkler score",
        "Winkler score",
        display_labels=display_labels,
        save_path=save_path,
    )
