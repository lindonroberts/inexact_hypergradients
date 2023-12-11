#!/usr/bin/env python3

"""
Do all data hypercleaning runs
"""
import os
import subprocess


def call_example(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize):
    cmd = "python data_hypercleaning_example.py "
    cmd += "--total_budget %g " % total_budget
    cmd += "--lower_solver %s " % lower_solver
    cmd += "--lower_tol %g " % lower_tol
    cmd += "--ad_method %s " % ad_method
    cmd += "--ad_tol %g " % ad_tol
    cmd += "--upper_stepsize %g " % upper_stepsize
    subprocess.run(cmd, shell=True)
    return


def files_exist(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize):
    results_path = 'raw_hypercleaning_results'
    if not os.path.isdir(results_path):  # depends on root directory
        results_path = os.path.join('ad_testing_v2', 'raw_hypercleaning_results')
    filestem = f'budget_{total_budget}_lower_{lower_solver}_{lower_tol}_ad_{ad_method}_{ad_tol}_step_{upper_stepsize}'
    results_file = os.path.join(results_path, filestem + '.csv')
    weights_file = os.path.join(results_path, filestem + '_data_weights.pt')
    params_file = os.path.join(results_path, filestem + '_w.pt')
    return os.path.isfile(results_file) and os.path.isfile(weights_file) and os.path.isfile(params_file)


def main():
    # test_only = True  # simple test
    test_only = False  # do full runs

    # skip_existing = False  # override old results
    skip_existing = True  # skip existing runs

    if test_only:
        total_budget = 1500
        lower_solvers = ['hb']
        lower_tols = [1e-2]
        ad_methods = ['gd']
        ad_tols = [1e-2]
        upper_stepsizes = [1e-2]
    else:
        # Proper runs
        total_budget = 100000
        lower_solvers = ['gd', 'hb', 'fista']
        lower_tols = [1e-1, 1e-2]  # expand later if needed
        ad_methods = ['gd', 'hb', 'cg']
        ad_tols = [1e-1, 1e-2]
        # upper_stepsizes = [1e-1, 1e-2]  # expand later if needed
        # upper_stepsizes = [1, 1e-1]  # expand later if needed
        upper_stepsizes = [10]

    for lower_solver in lower_solvers:
        for lower_tol in lower_tols:
            for ad_method in ad_methods:
                for ad_tol in ad_tols:
                    for upper_stepsize in upper_stepsizes:
                        if skip_existing and files_exist(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize):
                            continue  # skip
                        else:
                            call_example(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize)
    print("Done")
    return


if __name__ == '__main__':
    main()
