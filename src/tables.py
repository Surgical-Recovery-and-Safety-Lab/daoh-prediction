"""Functions generating the LaTeX tables of the paper."""

import tomllib
from pathlib import Path
from typing import Literal

from medpipe import MedpipeRegressor

from src.constants import BIN_DIST_EXP_ORDER, MAPPERS_EXP_ORDER, N_BINS_EXP_ORDER
from src.helpers import (
    bin_intervals,
    collect_bin_edges,
    generate_patient_characteristics_table,
    load_cv_summary,
    save_table,
)


def generate_cv_table(
    results: dict[str, dict[str, tuple[float, float]]],
    order: list[str],
    display_labels: dict[str, str],
    save_path: str | Path,
) -> None:
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


def generate_result_table(
    results: dict[str, dict[str, tuple[float, float, float]]],
    order: list[str],
    display_labels: dict[str, str],
    save_path: str | Path,
) -> None:
    """Generate the LaTeX metrics table with 95% CIs and save it.

    Parameters
    ----------
    results : dict[str, dict[str, tuple[float, float, float]]]
        Model key -> metrics (keys "mae", "maess", "crps" and "crpss"), each
        as (estimate, lower, upper). The CRPS and skill
        scores come from `compute_metrics`, and the MAE of the distributional
        models from `median_mae`. The MAE of the point-forecast models comes
        from `load_eval_results`. A
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


def generate_bin_edges_table(
    pipes: list[MedpipeRegressor],
    order: list[str],
    display_labels: dict[str, str],
    save_path: str | Path,
    per_line: int = 4,
) -> None:
    """Generate the LaTeX table of fitted bin edges and save it.

    One row per model, listing its integer bin intervals (see `bin_intervals`)
    wrapped over several lines.

    Parameters
    ----------
    pipes : list[MedpipeRegressor]
        Fitted pipelines, read through `pipe.models["DAOH_90"].bin_edges_`.
    order : list[str]
        Model keys (project names) controlling the row order.
    display_labels : dict[str, str]
        Model key -> LaTeX-escaped display string.
    save_path : str or Path
        File to save the table in.
    per_line : int, default=4
        Maximum number of intervals per line.

    """
    edges = collect_bin_edges(pipes)

    lines = [
        r"\begin{tabular}{cc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Bins} \\",
        r"\midrule",
    ]

    for key in order:
        intervals = bin_intervals(edges[key])
        chunks = [
            ", ".join(intervals[i : i + per_line])
            for i in range(0, len(intervals), per_line)
        ]
        for j, chunk in enumerate(chunks):
            last = j == len(chunks) - 1
            label = display_labels[key] if j == 0 else ""
            suffix = r" \\" if last else r", \\"
            lines.append(f"{label} & {chunk}{suffix}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    save_table("\n".join(lines), save_path)


def generate_prediction_table(
    results: dict[str, dict[int, tuple]],
    interval: float,
    order: list[str],
    display_labels: dict[str, str],
    save_path: str | Path,
) -> None:
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
        r"\begin{tabular}{lcccc}",
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
