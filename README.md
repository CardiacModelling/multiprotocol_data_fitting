# Fitting real multiprotocol data
This repository contains the code necessary to reproduce the results and figures in [Evaluating the predictive accuracy of ion-channel models using data from multiple experimental designs](https://google.co.uk).

## Requirements
Running this code requires an installation of python 3 and the [markovmodels](https://github.com/CardiacModelling/MarkovModels) package. The code in this repository has been tested on MacOS where the requisite packages have been installed using [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).

## Installation

It is recommended to install libraries and run scripts in a virtual environment to avoid version conflicts between different projects. First, clone the repository.

```
git clone https://github.com/CardiacModelling/evaluating_predictive_accuracy
```

Then, ensure that your `pip` installation is up to date.
```
python3 -m pip install --upgrade pip
```

Then, install the required packages.
```
python3 -m pip install -r requirements.txt
```


## Running
Fitting was performed using `scripts/fit_all_wells_and_protocols.py` which can be provided with the  `-w`, `--sweeps` `--protocols` command to select specific data traces to fit. This fitting is performed on data postprocessed by the [pcpostprocess](https://github.com/CardiacModelling/pcpostprocess) package.

The list of figures in the paper and the scripts that are run to produce them are shown in the table below. Some of these figures were produced using `pcpostprocess`.

| figure | script                                      | filename                                              |
|--------|---------------------------------------------|-------------------------------------------------------|
| 1      | NA                                          | NA                                                    |
| 2      | NA                                          | NA                                                    |
| 3      | plot_protocols.py                           | protocols_figure.pdf                                  |
| 4      | NA                                          | NA                                                    |
| 5      | optimisation_results.py                     | B20_staircaseramp1_sweep0.pdf                         |
| 6      | prediction_comparison.py                    | prediction_comparison.pdf                             |
| 7      | t_test_plots.py                             | average_sweep_0_t_scores_model3_0c_fitting.pdf        |
| 8      | t_test_plots.py                             | average_sweep_0_t_scores_model3_0c_prediction.pdf     |
| 9      | heatmaps.py                                 | best_worst_0c_model3_heatmap.pdf                      |
| 10     | heatmaps.py                                 | Case0c_heatmap_comparison.pdf                         |
| 11     | scatterplots.py                             | per_well_p1_p2_d_1.pdf                                |
| A.1    | pcpostprocess.scripts.run_herg_qc           | B20_staircaseramp1_before0.pdf                        |
| A.2    | pcpostprocess.scripts.run_herg_qc           | 25112022_MW-staircaseramp1-B20-sweep1-subtraction.pdf |
| D.1    | pcpostprocess.scripts.summarise_herg_export | E_rev.pdf                                             |
| D.2    | heatmaps.py                                 | averaged_well_heatmaps.pdf                            |
| E.1    | t_tests_plots.py                            | average_sweep0_t_scores_model2_0c_fitting.pdf         |
| E.2    | t_test_plots.py                             | average_sweep0_t_scores_model10_0c_fitting.pdf        |
| E.3    | t_test_plots.py                             | average_sweep0_t_scores_Wang_0c_fitting.pdf           |
| E.4    | t_test_plots.py                             | average_sweep0_t_scores_model2_0c_prediction.pdf      |
| E.5    | t_test_plots.py                             | average_sweep0_t_scores_model10_0c_prediction.pdf     |
| E.6    | t_test_plots.py                             | average_sweep0_t_scores_Wang_0c_prediction.pdf        |

