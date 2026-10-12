import json
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from medpipe import MedpipeRegressor
from medpipe.visualisation import themes
from ordboost.mappers import BaseBinMapper

from src.helpers import (
    compute_metrics,
    compute_sample_cdfs,
    coverage_sharpness_curve,
    estimate_patients,
    generate_patient_characteristics_table,
    marginal_calibration,
    pit_histogram,
    save_table,
    wrap_ngboost_pred_dist,
)
from src.plotting import (
    plot_binned_vs_continuous,
    plot_coverage,
    plot_marginal_calibration,
    plot_pit_histogram_grouped,
    plot_sample_cdfs,
    plot_sharpness,
    plot_winkler,
)

THEME = themes.MedpipeTheme()
N_BINS_EXP_ORDER = ["4-bins", "8-bins", "12-bins", "16-bins", "20-bins"]
BIN_DIST_EXP_ORDER = ["uniform", "quantile", "custom"]
MAPPERS_EXP_ORDER = ["mean", "median", "uniform", "continuous", "quantile"]


def generate_results(
    ordboost_version: str,
    ngboost_version: str,
    HGBr_version: str,
    mord_version: str,
) -> None:
    """Main result generating function saved in the ordboost artifact directory.

    Parameters
    ----------
    ordboost_version : str
        Version of the ordboost model to load.
    ngboost_version : str
        Version of the ngboost model to load.
    HGBr_version : str
        Version of the HGBr model to load.
    mord_version : str
        Version of the mord model to load.

    """
    ordboost_pipe = MedpipeRegressor.load(f"artifacts/ordboost/{ordboost_version}")
    ngboost_pipe = MedpipeRegressor.load(f"artifacts/ngboost/{ngboost_version}")
    hgbr_pipe = MedpipeRegressor.load(f"artifacts/HGBr/{HGBr_version}")
    mord_pipe = MedpipeRegressor.load(f"artifacts/mord/{mord_version}")

    ordboost_pipe._orchestrator.prepare_data()
    mord_pipe._orchestrator.prepare_data()

    (ordboost_pipe.run_dir / "results").mkdir(parents=True, exist_ok=True)
    (mord_pipe.run_dir / "results").mkdir(parents=True, exist_ok=True)
    (ordboost_pipe.run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (ngboost_pipe.run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Extract the data
    X_test = ordboost_pipe.data_split.X_test
    y_test = ordboost_pipe.data_split.y_test.to_numpy().squeeze()
    y_train = ordboost_pipe.data_split.y_train.to_numpy().squeeze()

    # Create the CDF distributions and predictions for other models
    y_preds = {}
    ngboost_dist = wrap_ngboost_pred_dist(
        ngboost_pipe.models["DAOH_90"].predict_dist(X_test)
    )
    ordboost_dist = ordboost_pipe.models["DAOH_90"].predict_dist(X_test)
    y_preds["HGBr"] = hgbr_pipe.predict(X_test, outcome="DAOH_90")
    y_preds["mord"] = mord_pipe.predict(
        mord_pipe.data_split.X_test,
        outcome="DAOH_90",
    )

    models = {"ngboost": ngboost_dist, "ordboost": ordboost_dist}
    table_models = models | {"HGBr": None, "mord": None}
    display_labels = {
        "ngboost": "NGBoost",
        "ordboost": "OrdBoost",
        "HGBr": "HGBregressor",
        "mord": "OrdinalRidge",
    }
    mapper = ordboost_pipe.models["DAOH_90"]["regressor"].mapper_

    # Generate data tables
    generate_data_table(ordboost_pipe)

    # Generate the data distribution figure (fig 1)
    generate_data_distribution(
        y_test,
        ordboost_pipe,
        mapper,
    )

    order = ["HGBr", "mord", "ngboost", "ordboost"]
    versions = {
        "HGBr": HGBr_version,
        "mord": mord_version,
        "ngboost": ngboost_version,
        "ordboost": ordboost_version,
    }
    pipes = {
        "HGBr": hgbr_pipe,
        "mord": mord_pipe,
        "ngboost": ngboost_pipe,
        "ordboost": ordboost_pipe,
    }
    results = {}
    for model, dist in table_models.items():
        y_true = y_test
        if model == "mord":
            y_true = mord_pipe.data_split.y_test["DAOH_90"].to_numpy()

        if dist is not None:
            y_pred = np.round(dist.median())
        else:
            y_pred = y_preds[model]

        # MAE and CRPS come from the evaluation, the skill scores are computed
        metrics = compute_metrics(y_true, y_pred, dist, y_train)
        results[model] = load_eval_results(f"artifacts/{model}/{versions[model]}") | {
            "maess": metrics["maess"],
            "crpss": metrics["crpss"],
        }

    generate_result_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "results/table.txt",
    )

    grid_y = ordboost_dist.grid_y  # Extracted for NGBoost to match
    indices = {298960: 0, 340896: 20, 26586: 70, 101: 85, 20: 89}

    results = {}
    for model, dist in models.items():
        results[model] = compute_sample_cdfs(dist, indices, grid_y)

    plot_sample_cdfs(
        results["ordboost"],
        colors=THEME.palette,
        title=display_labels["ordboost"],
        save_path=ordboost_pipe.run_dir / "plots/example_cdfs.png",
    )
    plot_sample_cdfs(
        results["ngboost"],
        colors=THEME.palette,
        title=display_labels["ngboost"],
        save_path=ngboost_pipe.run_dir / "plots/example_cdfs.png",
    )

    results = {}
    alpha = 0.05
    for model, dist in table_models.items():
        if dist is not None:
            results[model] = estimate_patients(dist, indices, alpha)
            continue
        res = {}
        for ind, daoh in indices.items():
            res[ind] = (daoh, y_preds[model][ind], None, None)
        results[model] = res

    generate_prediction_table(
        results,
        interval=(1 - alpha) * 100,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "results/prediction_table.txt",
    )

    # Redefine order and models without HGBr and mord
    order = ["ngboost", "ordboost"]
    display_labels = {"ngboost": "NGBoost", "ordboost": "OrdBoost"}
    results = {}
    for model, dist in models.items():
        results[model] = pit_histogram(y_test, dist, mapper)

    plot_pit_histogram_grouped(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/pit_histogram.png",
    )

    results = {}
    for model, dist in models.items():
        results[model] = marginal_calibration(y_test, dist, grid_y, mapper=mapper)

    plot_marginal_calibration(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/marginal_calibration.png",
    )

    coverage_levels = np.arange(10, 96, 5)  # 10, 15, ..., 90

    results = {}
    for model, dist in models.items():
        results[model] = coverage_sharpness_curve(y_test, dist, coverage_levels)

    plot_coverage(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/coverage.png",
    )

    plot_sharpness(
        results,
        order=order,
        display_labels=display_labels,
        save_path=ordboost_pipe.run_dir / "plots/sharpness.png",
    )

    generate_cv_results(ordboost_version, ngboost_version, HGBr_version, mord_version)


def generate_experiment_results(
    experiment: Literal["n_bins", "bin_dist", "mappers"],
) -> None:
    """Experiment result generating function.

    The results are saved in the experiments/{experiment} plots and
    results folders.

    Parameters
    ----------
    experiment : str, {"n_bins", "bin_dist", "mappers"}
        Experiment to generate results for.

    Raises
    ------
    FileNotFoundError
        If the artifacts directory or the subfolders are not found.

    """
    # Check that the artifacts folder exists and has 5 subfolders
    src_dir = Path(f"experiments/{experiment}")
    artifacts_dir = src_dir / "artifacts"

    n_artifacts = 3 if experiment == "bin_dist" else 5

    if not artifacts_dir.is_dir():
        msg = f"Missing directory: {artifacts_dir}"
        raise FileNotFoundError(msg)

    expected = {f"v{i}" for i in range(1, n_artifacts + 1)}
    found = {p.name for p in artifacts_dir.iterdir() if p.is_dir()}
    if missing := expected - found:
        msg = (
            f"{artifacts_dir} is missing subfolders: {sorted(missing)}. "
            "Run the experiment command with the --run flag first."
        )
        raise FileNotFoundError(msg)

    pipes = [
        MedpipeRegressor.load(artifacts_dir / f"v{i}")
        for i in range(1, n_artifacts + 1)
    ]

    # Extract the data
    pipes[0]._orchestrator.prepare_data()
    X_test = pipes[0].data_split.X_test
    y_test = pipes[0].data_split.y_test.to_numpy().squeeze()
    y_train = pipes[0].data_split.y_train.to_numpy().squeeze()

    models = {}
    mappers = {}
    for i, pipe in enumerate(pipes):
        name = pipe.mp_config.meta.project_name
        dist = pipe.models["DAOH_90"].predict_dist(X_test)

        models[name] = dist
        mappers[name] = pipe.models["DAOH_90"]["regressor"].mapper_

    display_labels = {
        "4-bins": "4-bins",
        "8-bins": "8-bins",
        "12-bins": "12-bins",
        "16-bins": "16-bins",
        "20-bins": "20-bins",
        "quantile": "Quantile",
        "uniform": "Uniform",
        "ordboost": "OrdBoost",
        "custom": "Custom",
        "mean": "Mean",
        "median": "Median",
        "continuous": "Continuous",
    }

    results = {}
    if experiment == "n_bins":
        order = N_BINS_EXP_ORDER
    elif experiment == "bin_dist":
        order = BIN_DIST_EXP_ORDER
    else:
        order = MAPPERS_EXP_ORDER

    for i, pipe in enumerate(pipes, start=1):
        name = pipe.mp_config.meta.project_name
        # MAE and CRPS come from the evaluation, the skill scores are computed
        metrics = compute_metrics(
            y_test,
            np.round(models[name].median()),
            models[name],
            y_train,
        )
        results[name] = load_eval_results(artifacts_dir / f"v{i}") | {
            "maess": metrics["maess"],
            "crpss": metrics["crpss"],
        }

    generate_result_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "results/table.txt",
    )

    results = {}
    for model, dist in models.items():
        mapper = mappers[model]
        results[model] = pit_histogram(y_test, dist, mapper)

    plot_pit_histogram_grouped(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "plots/pit_histogram.png",
    )

    results = {}
    for model, dist in models.items():
        grid_y = dist.grid_y
        mapper = mappers[model]
        results[model] = marginal_calibration(y_test, dist, grid_y, mapper=mapper)

    plot_marginal_calibration(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "plots/marginal_calibration.png",
    )

    coverage_levels = np.arange(10, 96, 5)  # 10, 15, ..., 90

    results = {}
    for model, dist in models.items():
        results[model] = coverage_sharpness_curve(y_test, dist, coverage_levels)

    plot_winkler(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "plots/winkler.png",
    )
    plot_coverage(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "plots/coverage.png",
    )

    plot_sharpness(
        results,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "plots/sharpness.png",
    )

    generate_experiment_cv_results(experiment)


def load_cv_summary(artifact_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Load the DAOH_90 cross-validation summary of an artifact.

    The scores are stored as negated losses (sklearn convention), so the sign
    is flipped to report MAE and CRPS as positive values. When the model has
    no CRPS (e.g. mord and HGBr) the CRPS is the MAE.

    Parameters
    ----------
    artifact_dir : str or Path
        Artifact directory containing the CV folder (e.g. artifacts/ordboost/v1).

    Returns
    -------
    dict[str, tuple[float, float]]
        Metric ("mae", "crps") -> (mean, std).

    Raises
    ------
    FileNotFoundError
        If the CV summary file does not exist.

    """
    path = Path(artifact_dir) / "CV" / "DAOH_90_cv_summary.json"
    if not path.is_file():
        msg = f"Missing CV summary: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        summary = json.load(f)

    results = {
        metric: (-summary[metric]["mean"], summary[metric]["std"])
        for metric in ("mae", "crps")
        if metric in summary
    }
    results.setdefault("crps", results["mae"])
    return results


def generate_cv_table(results, order, display_labels, save_path):
    """Generate the LaTeX cross-validation table (mean +/- std) and save it.

    Parameters
    ----------
    results : dict[str, dict[str, tuple[float, float]]]
        Model key -> metrics from `load_cv_summary`.
    order : list[str]
        Model keys controlling the column order.
    display_labels : dict[str, str]
        Model key -> LaTeX-escaped display string.
    save_path : str or Path
        File to save the table in.

    """
    lines = [
        r"\begin{tabular}{" + "c" * (len(order) + 1) + "}",
        r"\toprule",
        r"\textbf{Metrics} & "
        + " & ".join(rf"\textbf{{{display_labels[key]}}}" for key in order)
        + r" \\",
        r"\midrule",
    ]

    for row_label, metric_key in [("MAE (days)", "mae"), ("CRPS (days)", "crps")]:
        cells = []
        for key in order:
            mean, std = results[key][metric_key]
            cells.append(rf"${mean:.2f} \pm {std:.2f}$")
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    save_table("\n".join(lines), save_path)


def generate_cv_results(
    ordboost_version: str,
    ngboost_version: str,
    HGBr_version: str,
    mord_version: str,
) -> None:
    """Generate the cross-validation table for the main models.

    The table is saved in the ordboost artifact results folder.

    Parameters
    ----------
    ordboost_version : str
        Version of the ordboost model.
    ngboost_version : str
        Version of the ngboost model.
    HGBr_version : str
        Version of the HGBr model.
    mord_version : str
        Version of the mord model.

    """
    order = ["HGBr", "mord", "ngboost", "ordboost"]
    display_labels = {
        "ngboost": "NGBoost",
        "ordboost": "OrdBoost",
        "HGBr": "HGBregressor",
        "mord": "OrdinalRidge",
    }
    dirs = {
        "HGBr": Path(f"artifacts/HGBr/{HGBr_version}"),
        "mord": Path(f"artifacts/mord/{mord_version}"),
        "ngboost": Path(f"artifacts/ngboost/{ngboost_version}"),
        "ordboost": Path(f"artifacts/ordboost/{ordboost_version}"),
    }
    results = {model: load_cv_summary(dirs[model]) for model in order}

    save_dir = dirs["ordboost"] / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    generate_cv_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=save_dir / "cv_table.txt",
    )


def generate_experiment_cv_results(
    experiment: Literal["n_bins", "bin_dist", "mappers"],
) -> None:
    """Generate the cross-validation table for an experiment.

    Experiments are run in sorted configuration file order (v1, v2, ...), so
    each configuration's project name is used to find its artifact version and
    the columns are then arranged in the experiment's display order.

    Parameters
    ----------
    experiment : str, {"n_bins", "bin_dist", "mappers"}
        Experiment to generate the CV table for.

    Raises
    ------
    FileNotFoundError
        If a configuration or its CV summary is missing.
    ValueError
        If the configurations do not match the experiment's display order.

    """
    src_dir = Path(f"experiments/{experiment}")
    config_files = sorted(p for p in (src_dir / "config").glob("*.toml") if p.is_file())

    # Same sorted order as main.run_experiment: i-th config -> v{i}
    results = {}
    for i, config_file in enumerate(config_files, start=1):
        with open(config_file, "rb") as f:
            name = tomllib.load(f)["meta"]["project_name"]
        results[name] = load_cv_summary(src_dir / "artifacts" / f"v{i}")

    order = {
        "n_bins": N_BINS_EXP_ORDER,
        "bin_dist": BIN_DIST_EXP_ORDER,
        "mappers": MAPPERS_EXP_ORDER,
    }[experiment]
    if set(order) != set(results):
        msg = f"Configuration names {sorted(results)} do not match {order}"
        raise ValueError(msg)

    display_labels = {
        "4-bins": "4-bins",
        "8-bins": "8-bins",
        "12-bins": "12-bins",
        "16-bins": "16-bins",
        "20-bins": "20-bins",
        "quantile": "Quantile",
        "uniform": "Uniform",
        "custom": "Custom",
        "mean": "Mean",
        "median": "Median",
        "continuous": "Continuous",
    }

    res_path = src_dir / "results"
    res_path.mkdir(exist_ok=True)
    generate_cv_table(
        results,
        order=order,
        display_labels=display_labels,
        save_path=res_path / "cv_table.txt",
    )


def load_eval_results(
    artifact_dir: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """Load the DAOH_90 test evaluation results of an artifact.

    When the model has no CRPS (e.g. mord and HGBr) the CRPS is the MAE.

    Parameters
    ----------
    artifact_dir : str or Path
        Artifact directory containing the results folder
        (e.g. artifacts/ordboost/v1).

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Metric ("mae", "crps") -> (estimate, lower, upper), for the metrics
        that were evaluated.

    Raises
    ------
    FileNotFoundError
        If the evaluation results file does not exist.

    """
    path = Path(artifact_dir) / "results" / "DAOH_90_evaluation_results.json"
    if not path.is_file():
        msg = f"Missing evaluation results: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        overall = json.load(f)["overall"]

    results = {
        metric: (res["point_estimate"], res["ci_lower"], res["ci_upper"])
        for metric, res in overall.items()
    }
    if "mae" in results:
        results.setdefault("crps", results["mae"])
    return results


def generate_result_table(results, order, display_labels, save_path):
    """Generate the LaTeX metrics table with 95% CIs and save it.

    Parameters
    ----------
    results : dict[str, dict[str, tuple[float, float, float]]]
        Model key -> metrics (keys "mae", "maess", "crps" and "crpss"), each
        as (estimate, lower, upper). MAE and CRPS come from
        `load_eval_results` and the skill scores from `compute_metrics`. A
        metric row is only written if every model in `order` has it.
    order : list[str]
        Model keys controlling the column order.
    display_labels : dict[str, str]
        Model key -> LaTeX-escaped display string.
    save_path : str or Path
        File to save the table in.

    """
    lines = [
        r"\begin{tabular}{" + "c" * (len(order) + 1) + "}",
        r"\toprule",
        r"\textbf{Metrics} & "
        + " & ".join(rf"\textbf{{{display_labels[key]}}}" for key in order)
        + r" \\",
        r"\midrule",
    ]

    metric_rows = [
        ("MAE (days)", "mae", "{:.2f}"),
        ("MAESS", "maess", "{:.3f}"),
        ("CRPS (days)", "crps", "{:.2f}"),
        ("CRPSS", "crpss", "{:.3f}"),
    ]

    for row_label, metric_key, fmt in metric_rows:
        if not all(metric_key in results[key] for key in order):
            continue
        cells = []
        for key in order:
            value, lower, upper = results[key][metric_key]
            cells.append(
                f"{fmt.format(value)} [{fmt.format(lower)}, {fmt.format(upper)}]"
            )
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)

    save_table(table, save_path)


def generate_prediction_table(results, interval, order, display_labels, save_path):
    """Generate the LaTeX table of example patient predictions and save it.

    Models without prediction intervals (lower bound of None) are shown as
    point estimates.

    Parameters
    ----------
    results : dict[str, dict[int, tuple]]
        Model key -> {patient index: (DAOH, pred, l_bound, u_bound)}.
    interval : float
        Nominal coverage of the prediction intervals, in percent.
    order : list[str]
        Model keys controlling the column order.
    display_labels : dict[str, str]
        Model key -> LaTeX-escaped display string.
    save_path : str or Path
        File to save the table in.

    """
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
    ]
    indices = results[order[0]].keys()

    # Header row 1: DAOH label + model names
    header_row_1 = [r"$\mathbf{DAOH_{90}}$"]
    for key in order:
        label = display_labels.get(key, key)
        header_row_1.append(rf"\textbf{{{label}}}")
    lines.append(" & ".join(header_row_1) + r" \\")

    # Header row 2: units + median/PI subheader
    header_row_2 = [r"days"]
    for model in order:
        if model == "HGBr" or model == "mord":
            header_row_2.append(r"point-estimate")
        else:
            header_row_2.append(rf"median [{interval:.0f}\% PI]")
    lines.append(" & ".join(header_row_2) + r" \\")

    lines.append(r"\midrule")

    for idx in indices:
        row_daoh = None
        row_cells = []
        for key in order:
            daoh, pred, lower, upper = results[key][idx]
            if row_daoh is None:
                row_daoh = daoh
            if lower is not None:  # Only need to check one here
                row_cells.append(f"{pred:.0f} [{lower:.0f}--{upper:.0f}]")
            else:
                row_cells.append(f"{pred:.0f}")

        lines.append(f"{row_daoh:.0f} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)
    save_table(table, save_path)


def generate_data_table(pipe: MedpipeRegressor) -> None:
    """Generate the patient characteristic table for data splits.

    Parameters
    ----------
    pipe : MedpipeRegressor
        MedpipeRegressor pipeline to use for loading splits and saving.

    """
    X_test = pipe.data_split.X_test
    X_train = pipe.data_split.X_train

    category_orders = {
        "TRAUMA": X_test["TRAUMA"].unique(),
        "PRIOR_CANCER": X_test["PRIOR_CANCER"].unique(),
        "SEX": X_test["SEX"].unique(),
        "ETHNICITY": X_test["ETHNICITY"].unique(),
    }

    latex_table = generate_patient_characteristics_table(
        X_train,
        X_test,
        continuous_features=["AGE"],
        categorical_features=[
            "SEX",
            "ETHNICITY",
            "ADMISSION_SOURCE",
            "ADMISSION_ACUITY",
            "CATEGORY_LEVEL_1",
            "CATEGORY_LEVEL_2",
            "TRAUMA",
            "PRIOR_CANCER",
            "ASA",
            "OP_SEVERITY",
        ],
        feature_labels={
            "ASA": "ASA score",
            "OP_SEVERITY": "Operation severity",
            "ETHNICITY": "Ethnicity",
            "ADMISSION_SOURCE": "Admission source",
            "ADMISSION_ACUITY": "Admission acuity",
            "TRAUMA": "Trauma",
            "SEX": "Sex at birth",
            "PRIOR_CANCER": "Prior cancer",
            "CATEGORY_LEVEL_1": "Specialty",
            "CATEGORY_LEVEL_2": "Sub-specialty",
        },
        category_orders=category_orders,
        caption="Patient characteristics by dataset split.",
        label="tab:patient_chars",
    )
    save_table(latex_table, pipe.run_dir / "results/table_patient_chars.txt")


def generate_data_distribution(
    y_test: np.ndarray, pipe: MedpipeRegressor, mapper: BaseBinMapper
) -> None:
    """Generate the data distribution before and after binning.

    Parameters
    ----------
    y_test : np.ndarray of shape (n_samples,)
        Test data to plot.
    pipe : MedpipeRegressor
        MedpipeRegressor pipeline to save the data.
    mapper : BaseBinMapper
        Mapper used to extract the binned version of the data.

    """
    bin_edges = np.array(mapper.bin_edges)
    y_binned = mapper.digitize(y_test, bin_edges)

    plot_bins = np.concatenate(([0, 1], bin_edges[1:]))
    plot_binned_vs_continuous(
        y_test,
        y_binned,
        plot_bins,
        THEME.palette,
        save_path=pipe.run_dir / "plots/data_distributions.png",
    )
