"""Functions generating the figures and tables of the paper."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from medpipe import MedpipeClassifier, MedpipeRegressor
from medpipe.visualisation import themes
from ordboost.mappers import BaseBinMapper

from src.helpers import (
    binarise_daoh,
    compute_baseline_crps,
    compute_classifier_metrics,
    compute_metrics,
    compute_sample_cdfs,
    coverage_sharpness_curve,
    estimate_patients,
    load_eval_results,
    marginal_calibration,
    median_mae,
    pit_histogram,
    wrap_ngboost_pred_dist,
)
from src.plotting import (
    plot_binned_vs_continuous,
    plot_coverage,
    plot_marginal_calibration,
    plot_pit_histogram_grouped,
    plot_predicted_distribution,
    plot_sample_cdfs,
    plot_sharpness,
    plot_winkler,
)
from src.tables import (
    generate_bin_edges_table,
    generate_classifier_table,
    generate_cv_results,
    generate_data_table,
    generate_experiment_cv_results,
    generate_prediction_table,
    generate_result_table,
)

THEME = themes.MedpipeTheme()

from src.constants import (
    BIN_DIST_EXP_ORDER,
    MAPPERS_EXP_ORDER,
    N_BINS_EXP_ORDER,
    THRESHOLDS,
)


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
    y_preds["HGBr"] = np.clip(
        hgbr_pipe.predict(X_test, outcome="DAOH_90"), a_min=0, a_max=89
    )
    y_preds["mord"] = np.clip(
        mord_pipe.predict(
            mord_pipe.data_split.X_test,
            outcome="DAOH_90",
        ),
        a_min=0,
        a_max=89,
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

    # Generate the distribution of the test set predictions of each model
    generate_prediction_distributions(
        {
            "ngboost": ngboost_dist,
            "ordboost": ordboost_dist,
            "HGBr": y_preds["HGBr"],
            "mord": y_preds["mord"],
        },
        pipes={
            "ngboost": ngboost_pipe,
            "ordboost": ordboost_pipe,
            "HGBr": hgbr_pipe,
            "mord": mord_pipe,
        },
        display_labels=display_labels,
    )

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
    # One baseline CRPS, computed once, shared by the CRPSS of every model
    baseline_crps = compute_baseline_crps(y_test, y_train)

    results = {}
    for model, dist in table_models.items():
        y_true = y_test
        if model == "mord":
            y_true = mord_pipe.data_split.y_test["DAOH_90"].to_numpy()

        if dist is not None:
            y_pred = np.round(dist.median())
        else:
            y_pred = y_preds[model]

        # The skill scores and the CRPS of the distributional models come from
        # the same computation, so that the CRPS and CRPSS use one CRPS value.
        # Their MAE is computed on the median.
        metrics = compute_metrics(y_true, y_pred, dist, baseline_crps)
        evaluated = load_eval_results(f"artifacts/{model}/{versions[model]}")
        if dist is not None:
            evaluated["mae"] = median_mae(y_true, dist)
            evaluated["crps"] = metrics["crps"]
        results[model] = evaluated | {
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

    baseline_crps = compute_baseline_crps(y_test, y_train)

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
        # The CRPS and the skill scores come from the same computation, and
        # the MAE is computed on the median
        metrics = compute_metrics(
            y_test,
            np.round(models[name].median()),
            models[name],
            baseline_crps,
        )
        evaluated = load_eval_results(artifacts_dir / f"v{i}")
        evaluated["mae"] = median_mae(y_test, models[name])
        evaluated["crps"] = metrics["crps"]
        results[name] = evaluated | {
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

    generate_bin_edges_table(
        pipes,
        order=order,
        display_labels=display_labels,
        save_path=src_dir / "results/table_bins.txt",
    )

    generate_experiment_cv_results(experiment)


def generate_classifier_results(
    ordboost_version: str = "v1",
    ngboost_version: str = "v1",
) -> None:
    """Classifier experiment result generating function.

    For each threshold, compares the HGB classifier trained on
    DAOH < threshold with the NGBoost and OrdBoost models, using their
    predicted P(DAOH < threshold), on the test set. For a threshold of 0 the
    event is DAOH = 0 (see `binarise_daoh`). One table per threshold
    is saved in experiments/classification/results/table_{threshold}.txt.

    The classifiers are loaded from experiments/classification/artifacts/v{i},
    where i is the 1-based position of the threshold in `THRESHOLDS`, i.e. the
    order in which `run_classification` fits them.

    Parameters
    ----------
    ordboost_version : str, default="v1"
        Version of the ordboost model to load.
    ngboost_version : str, default="v1"
        Version of the ngboost model to load.

    Raises
    ------
    FileNotFoundError
        If a classifier artifact directory is not found.

    """
    src_dir = Path("experiments/classification")
    artifacts_dir = src_dir / "artifacts"
    (src_dir / "results").mkdir(parents=True, exist_ok=True)

    ordboost_pipe = MedpipeRegressor.load(f"artifacts/ordboost/{ordboost_version}")
    ngboost_pipe = MedpipeRegressor.load(f"artifacts/ngboost/{ngboost_version}")
    ordboost_pipe._orchestrator.prepare_data()
    ngboost_pipe._orchestrator.prepare_data()

    ordboost_dist = ordboost_pipe.models["DAOH_90"].predict_dist(
        ordboost_pipe.data_split.X_test
    )
    ngboost_dist = wrap_ngboost_pred_dist(
        ngboost_pipe.models["DAOH_90"].predict_dist(ngboost_pipe.data_split.X_test)
    )
    y_test = ordboost_pipe.data_split.y_test.to_numpy().squeeze()
    y_train = ordboost_pipe.data_split.y_train.to_numpy().squeeze()

    order = ["HGBc", "ngboost", "ordboost"]
    display_labels = {
        "HGBc": "HGBclassifier",
        "ngboost": "NGBoost",
        "ordboost": "OrdBoost",
    }

    for i, threshold in enumerate(THRESHOLDS, start=1):
        run_dir = artifacts_dir / f"v{i}"
        if not run_dir.is_dir():
            msg = f"Missing directory: {run_dir}. Run the classification experiment first."
            raise FileNotFoundError(msg)

        clf_pipe = MedpipeClassifier.load(run_dir)
        clf_pipe._orchestrator.prepare_data()
        outcome = next(iter(clf_pipe.models))
        y_clf = clf_pipe.data_split.y_test.to_numpy().squeeze()
        if not np.array_equal(y_clf, y_test):
            msg = f"The test set of {run_dir} differs from the regressors' test set."
            raise ValueError(msg)

        y_bin = binarise_daoh(y_test, threshold)
        if y_bin.min() == y_bin.max():
            print(f"Skipping threshold {threshold}: a single class in the test set")
            continue

        # P(event) of each model; for the DAOH = 0 event, P(DAOH <= 0.5)
        upper = 0.5 if threshold == 0 else threshold
        probs = {
            "HGBc": np.asarray(
                clf_pipe.predict_proba(clf_pipe.data_split.X_test, outcome=outcome)
            )[:, 1],
            "ngboost": ngboost_dist.cdf(upper),
            "ordboost": ordboost_dist.cdf(upper),
        }
        # Reference model: the training prevalence for every sample
        ref_prob = float(np.mean(binarise_daoh(y_train, threshold)))

        results = {
            model: compute_classifier_metrics(y_bin, prob, ref_prob)
            for model, prob in probs.items()
        }

        generate_classifier_table(
            results,
            order=order,
            display_labels=display_labels,
            save_path=src_dir / f"results/table_{threshold}.txt",
        )


def generate_prediction_distributions(
    predictions: dict[str, Any],
    pipes: dict[str, MedpipeRegressor],
    display_labels: dict[str, str],
) -> None:
    """Plot the distribution of the test set predictions, one figure per model.

    Each figure is saved in the plots folder of the model's artifact. The
    point prediction of the distributional models is their rounded median,
    and that of the other models is the output of their `predict`.

    Parameters
    ----------
    predictions : dict[str, Any]
        Model key -> predictive distribution (NGBoost and OrdBoost, with a
        `median` method) or array of point predictions.
    pipes : dict[str, MedpipeRegressor]
        Model key -> fitted pipeline, used for the save directory.
    display_labels : dict[str, str]
        Model key -> display string.

    """
    for model, pred in predictions.items():
        y_pred = np.round(pred.median()) if hasattr(pred, "median") else pred

        plot_dir = pipes[model].run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_predicted_distribution(
            y_pred,
            color=THEME.palette[0],
            title=display_labels[model],
            save_path=plot_dir / "predicted_distribution.png",
        )
        plt.close(fig)


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

    plot_bins = np.concatenate(([0], bin_edges, [90]))
    plot_binned_vs_continuous(
        y_test,
        y_binned,
        plot_bins,
        THEME.palette,
        save_path=pipe.run_dir / "plots/data_distributions.png",
    )
