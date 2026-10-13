# OrdBoost for days alive and out of hospital

Code accompanying our submission to the **ML4H 2026** conference.

This work builds on [OrdBoost](https://pypi.org/project/ordboost/) and evaluates
its performance on a clinical target: the number of **days alive and out of
hospital at 90 days** (DAOH₉₀). OrdBoost is compared with NGBoost, a histogram
gradient boosting regressor (HGBr) and an ordinal ridge regression (mord).

Experiments are driven by configuration files and run with
[Medpipe](https://pypi.org/project/medpipe/), so every fitted model and result
can be reproduced from its `.toml` file.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data availability

The data used to train and test the model were from the New Zealand National
Minimum Dataset (NMDS), which can be obtained from the New Zealand Ministry of
Health National Collections team. Requests can be made by contacting
data-enquiries@health.govt.nz.

## Usage

Everything is run through `main.py`, which has three subcommands.

### Fit a model

```bash
python main.py run {ordboost,ngboost,HGBr,mord}
```

Fits the selected model using `config/DAOH_config_<model>.toml`. Artifacts are
saved in `artifacts/<model>/`, one `v<N>` folder per run.

### Generate the main results

```bash
python main.py results [ordboost-version] [ngboost-version] [HGBr-version] [mord-version]
```

Loads the fitted models and generates the paper's tables and figures. Each
version defaults to `v1`. All four models must be fitted first. Tables and plots
are saved in the `results` and `plots` folders of the OrdBoost artifact.

### Run an experiment

```bash
python main.py experiment {n_bins,bin_dist,mappers} --run
python main.py experiment {n_bins,bin_dist,mappers} --results
```

`--run` fits every configuration in `experiments/<experiment>/config/`, and
`--results` generates the tables and figures from the fitted models. Run
`--run` first.

There are three experiments, all on OrdBoost:

| Experiment  | What it evaluates                                                        |
| ----------- | ------------------------------------------------------------------------ |
| `n_bins`    | The number of bins used to discretise the target (4, 8, 12, 16 and 20).  |
| `bin_dist`  | The bin distribution (uniform, quantile and custom bin edges).           |
| `mappers`   | The mapper strategy (mean, median, uniform, continuous and quantile).    |

Results are saved in `experiments/<experiment>/results` and
`experiments/<experiment>/plots`.

## Repository structure

```
main.py             Command line entry point
config/             Medpipe configuration of the four main models
experiments/        Configurations, artifacts and results of the three experiments
artifacts/          Fitted models, metrics and results of the main models
src/
  helpers.py          Metrics (MAE, CRPS, skill scores), PIT, calibration, coverage
  helpers_ngboost.py  Adapter exposing NGBoost distributions through OrdBoost's interface
  bootstrap.py        Bootstrap confidence intervals
  tables.py           LaTeX table generation
  plotting.py         Figures
  res_gen.py          Orchestrates the results of the main models and experiments
```

Fitted model files (`models/`) are not tracked in git.
