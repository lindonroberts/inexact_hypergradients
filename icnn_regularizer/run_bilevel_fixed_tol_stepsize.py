#!/usr/bin/env python3

"""
Run upper_level_solver.py (bilevel tuning) for specific settings file, but doing a parameter sweep over
fixed FISTA/CG tolerance and GD stepsizes.
"""
import logging
import multiprocessing
from upper_level_solver import do_batch_solve, do_single_run
from run_tune_standard_regularizers import get_num_cores, get_all_json_files

SETTINGS_FOLDER = 'settings_gd'


def single_run(input_info):
    settings_file, idx, init_tol, init_stepsize = input_info

    # Fixed inputs
    output_folder = 'raw_results'
    # max_lower_iters_per_image = 1000000  # total FISTA/CG work = 1e6 * num_training_images
    max_lower_iters_per_image = 10000000  # total FISTA/CG work = 1e7 * num_training_images
    decreasing_tol_stepsize = False  # fixed tol/stepsize only
    max_flat_iters = 100  # terminate on slow objective decrease
    flat_thresh = 1e-5
    # max_wall_runtime_seconds = None  # no timeout
    # max_wall_runtime_seconds = 10  # test
    # max_wall_runtime_seconds = 28800  # 8hrs
    # max_wall_runtime_seconds = 43200  # 12hrs
    max_wall_runtime_seconds = 86400  # 24hrs
    verbose = True
    verbose_to_print = False  # save to log file
    # save_train_recons = False  # don't save recons at each iteration
    save_train_recons = True  # save recons at each iteration

    if verbose and not verbose_to_print:
        proc_name = multiprocessing.current_process().name
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', filename='bilevel_fixed_tol_%s.log' % proc_name)

    do_single_run(settings_file, max_lower_iters_per_image, idx, init_tol, init_stepsize, decreasing_tol_stepsize,
                   max_flat_iters, flat_thresh, settings_folder=SETTINGS_FOLDER, output_folder=output_folder,
                   verbose=verbose, verbose_to_print=verbose_to_print, max_wall_runtime_seconds=max_wall_runtime_seconds,
                  save_train_recons=save_train_recons)
    return


def main():
    all_settings_files = get_all_json_files(SETTINGS_FOLDER, must_contain='final')
    # exponent_list = [0, -1, -2, -3, -4, -5]
    # exponent_list = [0, -2, -4]
    exponent_list = [0, -1, -2, -3, -4]
    # init_tol_list = [10 ** t for t in [0, -1, -2, -3, -4, -5]]
    # init_stepsize_list = [10 ** t for t in [0, -1, -2, -3, -4, -5]]
    init_tol_list = [10 ** t for t in exponent_list]
    init_stepsize_list = [10 ** t for t in exponent_list]

    all_runs = []
    for settings_file in all_settings_files:
        idx = 0
        for init_tol in init_tol_list:
            for init_stepsize in init_stepsize_list:
                all_runs.append((settings_file, idx, init_tol, init_stepsize))
                idx += 1

    ncores_to_use = get_num_cores() - 1
    print("Using %g cores to run %g jobs" % (ncores_to_use, len(all_runs)))
    with multiprocessing.Pool(ncores_to_use) as p:
        p.map(single_run, all_runs)
    return


if __name__ == '__main__':
    main()
