#!/usr/bin/env python3

"""
Make plots for data hypercleaning problem

Main code: data_hypercleaning_example.py
Run tests: data_hypercleaning_run_all.py
"""
import matplotlib.pyplot as plt
import os
import pandas as pd


RESULTS_FOLDER = 'raw_hypercleaning_results'
PLOTS_FOLDER = 'hypercleaning_plots'
# PLOT_COLS = ['k', 'b', 'r', 'g', 'c', 'm']
# PLOT_COLS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
# PLOT_COLS = ['#d7301f', '#fc8d59', '#fdcc8a']  # GD, HB, FISTA
# PLOT_COLS = ['#e66101', '#b2abd2', '#5e3c99']  # red, light purple, dark purple
# PLOT_COLS = ['black', 'gray', 'darkgray']  # IMA paper
PLOT_COLS = ['C0', 'C1', 'C2']  # arxiv
PLOT_MKRS = ['o', '^', 's']
PLOT_LS = ['-', '--', '-.']


def get_filestem(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize):
    return f'budget_{total_budget}_lower_{lower_solver}_{lower_tol}_ad_{ad_method}_{ad_tol}_step_{upper_stepsize}'


def get_results(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize, results_path=RESULTS_FOLDER):
    filestem = get_filestem(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize)
    results_file = os.path.join(results_path, filestem + '.csv')
    weights_file = os.path.join(results_path, filestem + '_data_weights.pt')  # torch tensor (torch.load)
    params_file = os.path.join(results_path, filestem + '_w.pt')  # torch tensor (torch.load)
    return pd.DataFrame.from_csv(results_file)


def plot_method_vary(total_budget, lower_solvers, lower_tol, ad_methods ,ad_tol, upper_stepsize, outfile=None,
                     font_size='small', fmt='png'):

    if outfile is None:
        plt.figure(figsize=(10,6))
        plt.gcf()
        plt.subplot(1, 2, 1)
    else:
        plt.figure(1)
        plt.clf()

    for lower_solver, col, mkr in zip(lower_solvers, PLOT_COLS, PLOT_MKRS):
        for ad_method, ls in zip(ad_methods, PLOT_LS):
            try:
                df = get_results(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize)
                plt.plot(df['cum_lower_iters'] + df['cum_ad_iters'], df['upper_obj'], lw=2,
                         label='%s/%s' % (lower_solver, ad_method), color=col, linestyle=ls, marker=mkr, markevery=len(df['cum_lower_iters'])//8)
            except FileNotFoundError:
                print("No results available for lower_solver '%s' and ad_method '%s'" % (lower_solver, ad_method))
    plt.grid()
    if outfile is None:
        plt.xlabel('Total lower-level/AD work')
        plt.ylabel('Upper objective')
        plt.legend(loc='best')
    else:
        ax = plt.gca()
        ax.set_ylim(0.44, 0.58)  # fixed y-axis limits for all paper plots
        ax.set_xlabel("Total lower-level/AD work", fontsize=font_size)
        ax.set_ylabel('Upper objective', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.legend(loc='best', fontsize=font_size, fancybox=True)
        filename = os.path.join(PLOTS_FOLDER, outfile + '_by_work.%s' % fmt)
        plt.savefig(filename, bbox_inches='tight')
        print("Saved %s" % filename)

    if outfile is None:
        plt.subplot(1, 2, 2)
    else:
        plt.figure(1)
        plt.clf()
    for lower_solver, col, mkr in zip(lower_solvers, PLOT_COLS, PLOT_MKRS):
        for ad_method, ls in zip(ad_methods, PLOT_LS):
            try:
                df = get_results(total_budget, lower_solver, lower_tol, ad_method, ad_tol, upper_stepsize)
                plt.plot(df.index, df['upper_obj'], lw=2, label='%s/%s' % (lower_solver, ad_method), color=col, linestyle=ls, marker=mkr, markevery=len(df)//10)
            except FileNotFoundError:
                print("No results available for lower_solver '%s' and ad_method '%s'" % (lower_solver, ad_method))
    plt.grid()
    if outfile is None:
        plt.xlabel('Upper iterations')
        plt.ylabel('Upper objective')
        plt.legend(loc='best')
    else:
        ax = plt.gca()
        ax.set_ylim(0.44, 0.58)  # fixed y-axis limits for all paper plots
        ax.set_xlabel("Upper iterations", fontsize=font_size)
        ax.set_ylabel('Upper objective', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.legend(loc='best', fontsize=font_size, fancybox=True)
        filename = os.path.join(PLOTS_FOLDER, outfile + '_by_iter.%s' % fmt)
        plt.savefig(filename, bbox_inches='tight')
        print("Saved %s" % filename)

    if outfile is None:
        plt.show()
    # else:
    #     filename = os.path.join(PLOTS_FOLDER, outfile + '.%s' % fmt)
    #     plt.savefig(filename, bbox_inches='tight')
    #     print("Saved %s" % filename)
    return


def main():
    total_budget = 100000
    lower_solvers = ['gd', 'hb', 'fista']
    lower_tols = [1e-1, 1e-2]  # expand later if needed
    # ad_methods = ['gd', 'hb', 'cg']
    ad_methods = ['hb', 'cg']  # don't include GD in paper (clutters plot, not very good)
    # ad_niters = [100, 1000]
    ad_tols = [1e-1, 1e-2]
    upper_stepsizes = [10.0]  #, 1.0, 1e-1]  #, 1e-2]  # expand later if needed

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font_size = 'x-large'
    fmt = 'pdf'

    # Vary lower solver & AD method
    for lower_tol in lower_tols:
        for upper_stepsize in upper_stepsizes:
            for ad_tol in ad_tols:
                outfile = f'method_vary_budget_{total_budget}_lower_{lower_tol}_ad_{ad_tol}_step_{upper_stepsize}'
                plot_method_vary(total_budget, lower_solvers, lower_tol, ad_methods, ad_tol, upper_stepsize,
                                 outfile=outfile, font_size=font_size, fmt=fmt)

    print("Done")
    return


if __name__ == '__main__':
    main()
