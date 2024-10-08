import logging
import multiprocessing
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import regex as re
import seaborn as sns
import pandas as pd
import numpy as np
import markovmodels

from argparse import ArgumentParser
from markovmodels.model_generation import make_model_of_class
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.BeattieModel import BeattieModel
from markovmodels.fitting import compute_predictions_df, get_best_params

matplotlib.use('agg')
pool_kws = {'maxtasksperchild': 1}


def fit_func(protocol, well, model_class, default_parameters=None, E_rev=None,
             randomise_initial_guess=True, prefix='', sweep=None,
             output_dir=None, args=None, default_kinetic_parameters=None):

    assert args is not None

    if output_dir:
        this_output_dir = os.path.join(output_dir, f"{prefix}{protocol}_{well}_sweep{sweep}")
    else:
        this_output_dir = None

    _E_rev = not args.dont_infer_Erev

    if args.use_artefact_model and not args.data_label:
        data_label = 'before'
    else:
        data_label = args.data_label

    fix_parameters = []
    if args.use_artefact_model:
        fix_parameters = [-1, -2, -3, -4, -5, -6, -7, -8]
        fix_parameters = [i % len(default_parameters) for i in fix_parameters]

    res_df = markovmodels.fitting.fit_well_data(
        model_class, well, protocol,
        args.data_directory,
        args.max_iterations,
        tolerance=args.tolerance,
        output_dir=this_output_dir,
        default_parameters=default_parameters,
        removal_duration=args.removal_duration,
        repeats=args.repeats,
        infer_E_rev=not args.dont_infer_Erev,
        experiment_name=args.experiment_name,
        E_rev=E_rev,
        population_size=args.population_size,
        randomise_initial_guess=randomise_initial_guess,
        solver_type=args.solver_type,
        sweep=sweep,
        use_artefact_model=args.use_artefact_model,
        fix_parameters=fix_parameters,
        data_label=data_label,
        artefact_default_kinetic_parameters=default_kinetic_parameters
    )

    res_df['well'] = well
    res_df['protocol'] = protocol
    res_df['sweep'] = sweep if sweep is not None else -1

    print(res_df)
    return res_df


def main():
    parser = ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument("--population_size", type=int)
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--dont_randomise_initial_guess', action='store_true')
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, type=int)
    parser.add_argument('--chain_length', '-l', default=500, type=int)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--artefact_default_kinetic_param_file')
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--dont_infer_Erev', action='store_true')
    parser.add_argument('--solver_type', default='default')
    parser.add_argument('--selection_file')
    parser.add_argument('--ignore_protocols', nargs='+', default=[])
    parser.add_argument('--ignore_wells', nargs='+', default=[])
    parser.add_argument('--sweeps', nargs='+', type=int, default=[0])
    parser.add_argument('--use_artefact_model', action='store_true')
    parser.add_argument('--subtraction_df_file')
    parser.add_argument('--data_label', default='')
    parser.add_argument('--compute_predictions', action='store_true')
    parser.add_argument('--reversal', type=float)
    parser.add_argument('--tolerance', nargs=2, type=float, default=(1e-8, 1e-8))
    parser.add_argument('-o', '--output')

    global args
    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    if args.selection_file:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
    else:
        selected_wells = None

    global default_artefact_kinetic_parameters
    default_artefact_kinetic_parameters = np.loadtxt(args.artefact_default_kinetic_param_file).flatten().astype(np.float64) if args.artefact_default_kinetic_param_file else None

    if args.use_artefact_model and not args.subtraction_df_file:
        raise Exception('Cannot use artefact model without qc file')

    if args.use_artefact_model and not args.reversal:
        raise Exception('Nernst potential must be provided when using an artefact model')

    if args.subtraction_df_file:
        subtraction_df = pd.read_csv(args.subtraction_df_file)

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(
        args.output,
        f"fitting_{args.experiment_name}_{args.model}"
    )

    print(f"outputting to {output_dir}")

    args.sweeps = list(set(args.sweeps))

    if len(args.wells) == 0:
        # If no wells are specified, use all wells
        args.wells = \
            all_wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16]
                         for i in range(1, 25)]

    protocols = args.protocols

    if args.selection_file:
        args.wells = [well for well in args.wells if well in selected_wells]

    if args.use_parameter_file:
        # Here we can use previous results to refit. Just use the best
        # parameters for each validation protocol as the initial guess
        best_params_df = pd.read_csv(args.use_parameter_file)
        if 'validation_protocol' in best_params_df:
            protocol_label = 'validation_protocol'
        else:
            protocol_label = 'protocol'

        best_params_df = get_best_params(best_params_df,
                                         protocol_label=protocol_label)
        assert(args.dont_randomise_initial_guess)

    else:
        best_params_df = None

    tasks = []
    protocols_list = []

    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9|_]*)-([A-Z|0-9]*)-sweep([0-9])-subtracted.csv$")

    subtraction_df['sweep'] = [max(0, sweep) for sweep in subtraction_df.sweep]

    param_labels = make_model_of_class(args.model).get_parameter_labels()
    for f in filter(regex.match, os.listdir(args.data_directory)):
        groups = re.search(regex, f).groups()
        protocol = groups[0]
        well = groups[1]
        if (protocols and protocol not in protocols) or well not in args.wells:
            continue
        if protocol in args.ignore_protocols or well in args.ignore_wells:
            continue

        if best_params_df is not None:
            parameter_row = best_params_df[(best_params_df.well.astype(str) == str(well))
                                           & (best_params_df[protocol_label] == protocol)].head(1)
            starting_parameters = parameter_row[param_labels].values.flatten().astype(np.float64)
        else:
            starting_parameters = None

        if not args.refit:
            prefix = ''
        else:
            prefix = 'prelim_'

        sweep = int(groups[2])

        # sweep can't be negative
        sweep = max(0, sweep)

        if args.sweeps:
            if int(sweep) not in args.sweeps:
                continue

        if args.use_artefact_model:
            row = subtraction_df[(subtraction_df.well == well) & (subtraction_df.protocol == protocol) &
                        (subtraction_df.sweep == sweep)]
            assert(row.shape[0] == 1)

            Rseries, Cm = row.iloc[0][['Rseries', 'Cm']]

            Rseries = Rseries * 1e-9
            Cm = Cm * 1e9

            gleak = 0.0
            Eleak = 0.0

            default_parameters = markovmodels.model_generation.make_model_of_class(args.model).get_default_parameters()
            starting_parameters = np.append(default_parameters,
                                            [args.reversal, gleak, Eleak, 0, 0, 0, Cm, Rseries]).astype(np.float64)

        tasks.append([protocol, well, args.model, starting_parameters, args.reversal,
                      not args.dont_randomise_initial_guess, prefix, sweep, output_dir, args,
                      default_artefact_kinetic_parameters])
        print(f"Tasks are {tasks}")

        protocols_list.append(protocol)

    assert len(tasks) > 0, "no valid protocol/well combinations provided"
    protocols_list = np.unique(protocols_list)
    pool_size = min(args.cores, len(tasks))

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    fitting_df = pd.concat(res, ignore_index=True)

    print("=============\nfinished fitting first round\n=============")

    fitting_df.to_csv(os.path.join(output_dir, f"{prefix}fitting.csv"))

    params_df = get_best_params(fitting_df)
    params_df.to_csv(os.path.join(output_dir, f"{prefix}best_fitting.csv"))

    if args.compute_predictions:
        predictions_df = compute_predictions_df(params_df, output_dir,
                                                f"{prefix}predictions",
                                                args=args,
                                                model_class=args.model,
                                                default_artefact_kinetic_parameters=default_artefact_kinetic_parameters)

        # Plot predictions
        predictions_df.to_csv(os.path.join(output_dir, f"{prefix}predictions_df.csv"))

        # Select best parameters for each protocol
        best_params_df_rows = []
        for well in predictions_df.well.unique():
            for validation_protocol in predictions_df['validation_protocol'].unique():
                sub_df = predictions_df[(predictions_df.validation_protocol ==
                                        validation_protocol) & (predictions_df.well == well)]

                best_param_row = sub_df[sub_df.score == sub_df['score'].min()].head(1).copy()
                best_params_df_rows.append(best_param_row)

        best_params_df = pd.concat(best_params_df_rows, ignore_index=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
