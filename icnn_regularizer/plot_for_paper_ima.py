#!/usr/bin/env python

"""
Plots for IMA paper (second version)

Sec 4.1. Simple quadratic problem (Figs 1-3): see quadratic_problem_example_v2.py - rerun as needed, very quick
Images saved in ./quadratic_problem_plots_v2

Sec 4.2. Data hypercleaning (Fig 4): see data_hypercleaning_plots.py
Based on runs from data_hypercleaning_example.py, called in data_hypercleaning_run_all.py (taks ~2 days to run)
Images saved in ./hypercleaning_plots

Sec 4.3. Convex NN: created in this file (based on plots_for_paper.py)
Images saved in ./paper_plots_v2
Note: reconstruction results are the same as plots_for_paper.py, so these files are just copied over to save time)
Plotting of reconstructions taken from plot_recons.py in old Overleaf document
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
try:
    from ad_testing_v2.utils import read_json, save_dict
except ModuleNotFoundError:
    from utils import read_json, save_dict
from run_tune_standard_regularizers import get_all_json_files
from plot_for_paper import save_single_bilevel_tv_comparison_recons

OUTFOLDER = 'paper_plots_v2'


def plot_bilevel_decrease(results, stepsize_to_use, tols_plot_info, font_size, by_runtime=False, fmt='eps', figsize=None):
    outfolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison')

    plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    raw_results = {}
    for tol, col, mkr in tols_plot_info:
        if (tol, stepsize_to_use) in results:
            mydict, df = results[(tol, stepsize_to_use)]
            xvals = df['wall_runtime'].to_numpy() if by_runtime else df['total_fista_cg_iters'].to_numpy()
            yvals = df['obj'].to_numpy()
            # if obj_best_so_far:
            #     yvals = np.minimum.accumulate(yvals)
            # Plot, with consistent formatting across different methods/iters
            lbl = 'Tol = %g' % tol
            plot_fun = plt.semilogy if by_runtime else plt.loglog  # use loglog for FISTA/AD iters plot
            plot_fun(xvals, yvals, color=col, marker=mkr, linewidth=2, label=lbl, markevery=100)
            raw_results[lbl] = {'xvals': [float(x) for x in xvals], 'yvals': [float(y) for y in yvals]}
    ax.set_xlabel("Runtime in seconds" if by_runtime else "Total FISTA/AD Iterations", fontsize=font_size)
    ax.set_ylabel("Upper-level objective", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()
    filename = 'bilevel_stepsize%g_obj_%s.%s' % (stepsize_to_use, 'by_runtime' if by_runtime else 'by_iters', fmt)
    plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
    save_dict(raw_results, os.path.join(outfolder, filename.replace('.%s' % fmt, '.json')))
    return


def plot_bilevel_tv_comparison(font_size, fmt='png'):
    # Plot comparison between good TV and long bilevel runs
    # Based on plot_standard_regularizers.py / run_tune_standard_regularizers.py for TV
    # and plot_bilevel_fixed_tol_stepsize.py / run_bilevel_fixed_tol_stepsize.py & run_bilevel_fixed_restarts.py
    outfolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison')

    # Get fixed tol/stepsize results
    print("Loading data")
    setting_name = 'denoising1d_pwlinear_shallow_final'
    this_results_folder = os.path.join('raw_results', setting_name)
    all_files = sorted(get_all_json_files(this_results_folder, must_contain='%s_run' % setting_name))
    all_files = [os.path.join(this_results_folder, f) for f in all_files]

    results = {}  # key = (tol, stepsize)
    for run_file in all_files:
        mydict = read_json(run_file)
        stepsize = mydict['gd_solver']['init_stepsize']
        tol = mydict['gd_solver']['init_tol']
        df = pd.read_csv(run_file.replace('.json', '.csv'))
        if len(df) == 0:
            print("Skipping empty csv file: %s" % run_file.replace('.json', '.csv'))
            continue
        df['total_fista_iters'] = df['fista_iters'].cumsum()
        df['total_cg_iters'] = df['cg_iters'].cumsum()
        df['total_fista_cg_iters'] = df['total_fista_iters'] + df['total_cg_iters']
        results[(tol, stepsize)] = (mydict, df)
    tols = sorted(list(set([k[0] for k in results.keys()])))[::-1]
    stepsizes = sorted(list(set([k[1] for k in results.keys()])))[::-1]

    # Plot objective decrease (one plot per choice of stepsize)
    print("Making objective decrease plots")
    stepsizes = [0.01]
    # tols_plot_info = [(0.01, '#d7301f', 'o'), (0.001, '#fc8d59', '^'), (0.0001, '#fdcc8a', 's')]  # just show the ones with very long runtimes
    # tols_plot_info = [(0.01, 'black', 'o'), (0.001, 'gray', '^'), (0.0001, 'darkgray', 's')]  # IMA paper
    tols_plot_info = [(0.01, 'C0', 'o'), (0.001, 'C1', '^'), (0.0001, 'C2', 's')]  # arxiv
    for stepsize in stepsizes:
        plot_bilevel_decrease(results, stepsize, tols_plot_info, font_size, by_runtime=False, fmt=fmt)
        plot_bilevel_decrease(results, stepsize, tols_plot_info, font_size, by_runtime=True, fmt=fmt)
        """
        Below is (very slow) code to produce reconstruction json files, one per tolerance
        In outfolder, these are copied from plots_for_paper.py outfolder results since they are the same
        Also copied tv reconstruction results with optimal parameter choice (plots_for_paper.py -> save_tv_recons)
        """
        # for tol in tols:
        #     print("Saving reconstruction data for stepsize = %g, tol = %g" % (stepsize, tol))
        #     recons_results = save_single_bilevel_tv_comparison_recons(results, stepsize, tol, recons_fista_xtol=1e-5)
        #     save_dict(recons_results, os.path.join(outfolder, 'bilevel_stepsize%g_tol%g_recons_data.json' % (stepsize, tol)))
    return


def read_recons_results(filename):
    results = read_json(filename)
    xvals = np.array(results['raw_dataset']['xvals'])
    all_training_ids = sorted([k for k in results['raw_dataset'].keys() if k.startswith('train')])
    all_test_ids = sorted([k for k in results['raw_dataset'].keys() if k.startswith('test')])

    training_data = {}
    training_data['xvals'] = xvals
    training_data['loss_mean'] = results['recons_train']['loss_mean']
    training_data['loss_vec'] = np.array(results['recons_train']['loss_vec'])
    training_data['recons_data'] = {}
    for id in all_training_ids:
        training_data['recons_data'][id] = {}
        training_data['recons_data'][id]['true_img'] = np.array(results['raw_dataset'][id]['true_img'])
        training_data['recons_data'][id]['noisy_data'] = np.array(results['raw_dataset'][id]['noisy_data'])
        training_data['recons_data'][id]['recons'] = np.array(results['recons_train'][id])

    test_data = {}
    test_data['xvals'] = xvals
    test_data['loss_mean'] = results['recons_test']['loss_mean']
    test_data['loss_vec'] = np.array(results['recons_test']['loss_vec'])
    test_data['recons_data'] = {}
    for id in all_test_ids:
        test_data['recons_data'][id] = {}
        test_data['recons_data'][id]['true_img'] = np.array(results['raw_dataset'][id]['true_img'])
        test_data['recons_data'][id]['noisy_data'] = np.array(results['raw_dataset'][id]['noisy_data'])
        test_data['recons_data'][id]['recons'] = np.array(results['recons_test'][id])
    return training_data, test_data


def plot_single_recons(xvals, true_img, noisy_data, recons, outfolder, filename, font_size='large', fmt='png'):
    # If recons is None, just show noisy & true image
    plt.figure()
    plt.clf()
    ax = plt.gca()
    plt.plot(xvals, true_img, ls='-', color='C0', linewidth=2.0, label='True data')  # arxiv
    plt.plot(xvals, noisy_data, ls='-', color='C1', linewidth=1.5, label='Noisy data')  # arxiv
    # plt.plot(xvals, true_img, ls='-', color='black', linewidth=1.5, label='True data')  # IMA paper
    # plt.plot(xvals, noisy_data, ls='--', color='darkgray', linewidth=1.5, label='Noisy data')  # IMA paper
    if recons is not None:
        plt.plot(xvals, recons, ls='-', color='C3', linewidth=2.0, label='Reconstruction')  # arxiv
        # plt.plot(xvals, recons, ls='-', color='gray', linewidth=2.0, label='Reconstruction', markevery=8, marker='o')  # IMA paper
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()
    plt.savefig(os.path.join(outfolder, '%s.%s' % (filename, fmt)), bbox_inches='tight')
    return


def plot_recons(infolder, infile, font_size='large', fmt='png'):
    training_data, test_data = read_recons_results(os.path.join(infolder, infile))

    # Where to save reconstruction plots?
    outfolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison_recons')
    plot_filename_stem = infile.replace('_recons_data.json', '')

    # Print loss values and show results
    print("Training dataset:")
    # print(" - Loss vector =", training_data['loss_vec'])
    print(" - Avg loss = %g" % training_data['loss_mean'])
    print(" - Plotting...")
    for id in training_data['recons_data'].keys():
        print(id)
        xvals = training_data['xvals']
        true_img = training_data['recons_data'][id]['true_img']
        noisy_data = training_data['recons_data'][id]['noisy_data']
        recons = training_data['recons_data'][id]['recons']  # change to None to just show dataset example
        filename = '%s_recons_%s' % (plot_filename_stem, id)
        plot_single_recons(xvals, true_img, noisy_data, recons, outfolder, filename, font_size=font_size, fmt=fmt)

    print("Test dataset:")
    # print(" - Loss vector =", test_data['loss_vec'])
    print(" - Avg loss = %g" % test_data['loss_mean'])
    print(" - Plotting...")
    for id in test_data['recons_data'].keys():
        print(id)
        xvals = test_data['xvals']
        true_img = test_data['recons_data'][id]['true_img']
        noisy_data = test_data['recons_data'][id]['noisy_data']
        recons = test_data['recons_data'][id]['recons']
        filename = '%s_recons_%s' % (plot_filename_stem, id)
        plot_single_recons(xvals, true_img, noisy_data, recons, outfolder, filename, font_size=font_size, fmt=fmt)
    return


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    font_size = 'x-large'
    fmt = 'pdf'
    plot_bilevel_tv_comparison(font_size=font_size, fmt=fmt)

    # Reconstruction results
    infolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison')
    infile_list = []
    infile_list.append('tv_recons_data.json')
    infile_list.append('bilevel_stepsize0.01_tol0.01_recons_data.json')  # has best results
    # infile_list.append('bilevel_stepsize0.01_tol0.001_recons_data.json')
    # infile_list.append('bilevel_stepsize0.01_tol0.0001_recons_data.json')
    for infile in infile_list:
        print("**** %s ****" % infile)
        plot_recons(infolder, infile, font_size=font_size, fmt=fmt)
        print("******************")
    print("Done")
    return


if __name__ == '__main__':
    main()
