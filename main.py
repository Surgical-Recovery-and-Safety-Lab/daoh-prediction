"""Command line entry point to fit the models and generate the results."""

import argparse
import sys
from pathlib import Path
from typing import Literal

from medpipe import MedpipeRegressor, MetricRegistry, ModelRegistry
from medpipe.metrics import MetricSpec
from mord import OrdinalRidge

from src.helpers_ngboost import crps_ngboost_per_sample, crps_ngboost_score
from src.res_gen import generate_experiment_results, generate_results


def run_model(model: Literal["ordboost", "ngboost", "HGBr", "mord"]) -> None:
    """Fit one of the available models.

    Parameters
    ----------
    model : str, {"ordboost", "ngboost", "HGBr", "mord"}
        Name of the model to run.

    """
    config = f"config/DAOH_config_{model}.toml"
    dir = f"artifacts/{model}"

    pipe = MedpipeRegressor(config=config, base_artifact_dir=dir)
    pipe.run()

    # Make directories to save results in case they don't exist
    plot_path = pipe.run_dir / "plots"
    plot_path.mkdir(exist_ok=True)
    res_path = pipe.run_dir / "results"
    res_path.mkdir(exist_ok=True)


def run_experiment(
    experiment: Literal["n_bins", "bin_dist", "mappers"],
    run_flag: Literal["run", "results"],
) -> None:
    """Run the selected experiment.

    Parameters
    ----------
    experiment : str, {"n_bins", "bin_dist", "mappers"}
        Selected experiment to run.
    run_flag : str, {"run", "results"}
        Flag to select how to run the experiment. The `run` mode fits all the
        models in the experiment. The `results` mode generates the results, if
        the models have been fitted first.

    """
    src_dir = Path(f"experiments/{experiment}")
    config_dir = src_dir / "config"  # Configuration file directory
    pattern = "*.toml"  # Get only .toml files

    config_files = sorted(p for p in config_dir.glob(pattern) if p.is_file())

    if run_flag == "results":
        # Make directories to save results in case they don't exist
        plot_path = src_dir / "plots"
        plot_path.mkdir(exist_ok=True)
        res_path = src_dir / "results"
        res_path.mkdir(exist_ok=True)

        generate_experiment_results(experiment)
        sys.exit()

    for config_file in config_files:
        print(f"Configuration file: {config_file}")
        pipe = MedpipeRegressor(
            config=config_file, base_artifact_dir=src_dir / "artifacts"
        )
        pipe.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main", description="Fit pipelines and generate figures for ml4h paper"
    )

    subparsers = parser.add_subparsers(prog="main", dest="command")

    # Main model fitting parser
    run_parser = subparsers.add_parser(
        "run", help="fit a model", description="Run one of the available models"
    )

    run_parser.add_argument(
        "model",
        help="model to select",
        choices=[
            "ordboost",
            "ngboost",
            "HGBr",
            "mord",
        ],
    )

    # Main results generation parser
    gen_parser = subparsers.add_parser(
        "results", help="generate results", description="Generate one or more figures"
    )
    gen_parser.add_argument(
        "ordboost_version",
        metavar="ordboost-version",
        help="version for the ordboost model (e.g. v1)",
        nargs="?",
        default="v1",
    )
    gen_parser.add_argument(
        "ngboost_version",
        metavar="ngboost-version",
        help="version for the ngboost model (e.g. v1)",
        nargs="?",
        default="v1",
    )
    gen_parser.add_argument(
        "hgbr_version",
        metavar="HGBr-version",
        help="version for the HGBr model (e.g. v1)",
        nargs="?",
        default="v1",
    )
    gen_parser.add_argument(
        "mord_version",
        metavar="mord-version",
        help="version for the mord model (e.g. v1)",
        nargs="?",
        default="v1",
    )

    # Experiment parser
    exp_parser = subparsers.add_parser(
        "experiment",
        help="run an experiment",
        description="Run one of the available experiments",
    )
    exp_parser.add_argument(
        "experiment",
        choices=["n_bins", "bin_dist", "mappers"],
        help="experiment to run",
    )
    exp_group = exp_parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        "--run",
        action="store_true",
        help="fit all the models from the selected experiment",
    )
    exp_group.add_argument(
        "--results",
        action="store_true",
        help="generate all the results for the selected experiment",
    )

    args = parser.parse_args()

    ModelRegistry.register(name="OrdinalRidge")(OrdinalRidge)

    MetricRegistry.register_spec(
        MetricSpec(
            name="crps_ngboost",
            func=crps_ngboost_score,
            response_method="predict_dist",
            display_name="CRPS (NGBoost)",
            per_sample_func=crps_ngboost_per_sample,
        )
    )

    if args.command == "run":
        # The run subcommand was called
        run_model(model=args.model)
    elif args.command == "results":
        # The results command was called
        generate_results(
            args.ordboost_version,
            args.ngboost_version,
            args.hgbr_version,
            args.mord_version,
        )

    else:
        # The experiment command was called
        if args.run:
            run_flag = "run"
        else:
            run_flag = "results"
        run_experiment(experiment=args.experiment, run_flag=run_flag)
