#!/usr/bin/env python3

"""
Make plots for NeurIPS 2022 paper
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
try:
    from ad_testing_v2.utils import estimate_fista_maxiters, fista, conjugate_gradient
    from ad_testing_v2.utils import read_json, save_dict, numpy_to_tensor, tv_denoise
except ModuleNotFoundError:
    from utils import estimate_fista_maxiters, fista, conjugate_gradient
    from utils import read_json, save_dict, numpy_to_tensor, tv_denoise
from upper_level_solver import get_model_and_dataset, get_final_params, get_final_recons, calculate_upper_obj_and_gradient
from run_tune_standard_regularizers import get_all_json_files
from plot_bilevel_fixed_tol_stepsize import get_best_params
from tune_standard_regularizers import generate_dataset

OUTFOLDER = 'paper_plots'


def cm2inch(*tupl):
    # pyplot figsize argument must be in inches, this does the conversion from cm
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_single_ad_cg_test(tol, niters, font_size, fmt='eps', figsize=None):
    # Plot of convergence in single gradient estimates of each AD method
    # Based on run_ad_cg_testing.py
    infolder = os.path.join('raw_results', 'ad_cg_testing')
    outfolder = os.path.join(OUTFOLDER, 'ad_cg_testing')
    row_norms = lambda A: np.linalg.norm(A, axis=1)

    all_results = read_json(os.path.join(infolder, 'ad_cg_test_tol%g_niters%g.json' % (tol, niters)))

    # Load gradient history
    methods = ['cg', 'ad_gd', 'ad_hb']
    nparams = len(all_results['model_params'])
    grad = {}
    for method in methods:
        grad_hist = np.zeros((niters+1, nparams))
        for i in range(nparams):
            grad_hist[:,i] = np.array(all_results['grad'][method]['param%g' % i])
        grad[method] = grad_hist

    # Plot convergence for each parameter separately
    labels = {'cg': 'IFT', 'ad_gd': 'GD', 'ad_hb': 'HB'}
    for i in range(nparams):
        raw_data = {}
        raw_data['Iterations'] = [int(x) for x in np.arange(niters+1)]
        plt.figure(figsize=figsize)
        plt.clf()
        ax = plt.gca()
        for method in grad:
            plt.semilogx(grad[method][:, i], linewidth=2, label=labels[method])
            raw_data[labels[method]] = [float(y) for y in grad[method][:,i]]
        ax.set_xlabel("Iterations", fontsize=font_size)
        ax.set_ylabel("Gradient component %g" % i, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.legend(loc='best', fontsize=font_size, fancybox=True)
        plt.grid()
        filename = 'tol%g_niters%g_param%g_raw.%s' % (tol, niters, i, fmt)
        plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
        df = pd.DataFrame.from_dict(raw_data)
        df.to_csv(os.path.join(outfolder, filename.replace('.%s' % fmt, '.csv')), index=False)

    # Plot convergence of gradient norm (doesn't imply convergence per-parameter, but simplifies the plot)
    plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    raw_data = {}
    raw_data['Iterations'] = [int(x) for x in np.arange(niters + 1)]
    for method in grad:
        plt.semilogx(row_norms(grad[method]), linewidth=2, label=labels[method])
        raw_data[labels[method]] = [float(y) for y in row_norms(grad[method])]
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Gradient norm", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()
    filename = 'tol%g_niters%g_full_raw.%s' % (tol, niters, fmt)
    plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
    df = pd.DataFrame.from_dict(raw_data)
    df.to_csv(os.path.join(outfolder, filename.replace('.%s' % fmt, '.csv')), index=False)

    # Plot error in gradient for each parameter separately
    true_grad = grad['cg'][-1,:]
    # for i in range(nparams):
    #     plt.figure(figsize=figsize)
    #     plt.clf()
    #     ax = plt.gca()
    #     for method in grad:
    #         plt.semilogy(np.abs(grad[method][:, i] - true_grad[i]), linewidth=2, label=labels[method])
    #     ax.set_xlabel("Iterations", fontsize=font_size)
    #     ax.set_ylabel("Abs error in gradient component %g" % i, fontsize=font_size)
    #     ax.tick_params(axis='both', which='major', labelsize=font_size)
    #     ax.legend(loc='best', fontsize=font_size, fancybox=True)
    #     plt.grid()
    #     filename = 'tol%g_niters%g_param%g_error.%s' % (tol, niters, i, fmt)
    #     plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')

    # Plot norm of error
    plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    raw_data = {}
    raw_data['Iterations'] = [int(x) for x in np.arange(niters + 1)]
    for method in grad:
        plt.semilogy(row_norms(grad[method] - true_grad), linewidth=2, label=labels[method])
        raw_data[labels[method]] = [float(y) for y in row_norms(grad[method] - true_grad)]
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Abs gradient error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()
    plt.gcf().tight_layout(pad=0.01)
    filename = 'tol%g_niters%g_full_error.%s' % (tol, niters, fmt)
    plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
    df = pd.DataFrame.from_dict(raw_data)
    df.to_csv(os.path.join(outfolder, filename.replace('.%s' % fmt, '.csv')), index=False)
    return


def plot_ad_cg_testing(font_size, fmt='eps'):
    # Plot of convergence in single gradient estimates of each AD method
    # Based on run_ad_cg_testing.py
    # tols = [1, 0, -1, -2, -5, -8]  # fista_xtol = 10**(tol)
    tols = [1, -2, -5]  # fista_xtol = 10**(tol)
    # tols = [1]
    niters = 1000
    for tol in tols:
        print("AD/CG testing for tol = %g" % tol)
        plot_single_ad_cg_test(tol, niters, font_size, fmt=fmt, figsize=cm2inch(6.1, 4.1))
    return


def parse_ad_method_iters(ad_method_iters):
    # ad_method_iters = cg_100 or ad_hb_10, etc.
    # Get the method and iters separately
    split_idx = ad_method_iters.rindex('_')
    method = ad_method_iters[:split_idx]
    niters = int(ad_method_iters[split_idx + 1:])
    return method, niters


def plot_single_ad_cg_bilevel_obj_dec(results, ad_method_iters_list, tol, font_size, by_runtime=False, fmt='eps', figsize=None):
    PLOT_COL = {'cg': 'C0', 'ad_gd': 'C1', 'ad_hb': 'C2'}  # per method
    PLOT_FMT = {10: '-', 100: '--', 1000: '-.'}  # per niters
    labels = {'cg': 'IFT-[IT]', 'ad_gd': 'GD-[IT]', 'ad_hb': 'HB-[IT]'}
    outfolder = os.path.join(OUTFOLDER, 'ad_cg_bilevel')

    plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    raw_results = {}
    for ad_method_iters in ad_method_iters_list:
        if (tol, ad_method_iters) in results:
            mydict, df = results[(tol, ad_method_iters)]
            xvals = df['wall_runtime'].to_numpy() if by_runtime else df['total_fista_cg_iters'].to_numpy()
            yvals = df['obj'].to_numpy()
            # if obj_best_so_far:
            #     yvals = np.minimum.accumulate(yvals)
            # Plot, with consistent formatting across different methods/iters
            method, niters = parse_ad_method_iters(ad_method_iters)
            lbl = labels[method].replace('[IT]', '%s' % niters)
            plot_fun = plt.semilogy if by_runtime else plt.loglog  # use loglog for FISTA/AD iters plot
            plot_fun(xvals, yvals, linewidth=2, label=lbl, color=PLOT_COL[method], linestyle=PLOT_FMT[niters])
            raw_results[lbl] = {'xvals': [float(x) for x in xvals], 'yvals': [float(y) for y in yvals]}
    ax.set_xlabel("Runtime in seconds" if by_runtime else "Total FISTA/AD Iterations", fontsize=font_size)
    ax.set_ylabel("Upper-level objective", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()
    filename = 'bilevel_run_tol%g_obj_%s.%s' % (tol, 'by_runtime' if by_runtime else 'by_iters', fmt)
    plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
    save_dict(raw_results, os.path.join(outfolder, filename.replace('.%s' % fmt, '.json')))
    return


def save_single_ad_cg_bilevel_recons(results, ad_method_iters, tol, recons_fista_xtol=1e-5):
    best_params = get_best_params(results)
    mydict, df = results[(tol, ad_method_iters)]

    model, dataset = get_model_and_dataset(mydict)
    p = model.parameter_values()
    pnew = torch.tensor(best_params[(tol, ad_method_iters)]).reshape(p.shape)
    model.set_params(pnew)

    recons_results = {}
    recons_results['tol'] = tol
    recons_results['ad_method_iters'] = ad_method_iters
    recons_results['recons_fista_xtol'] = recons_fista_xtol
    recons_results['model_params'] = [float(p) for p in best_params[(tol, ad_method_iters)].flatten()]

    # Save raw data
    recons_results['raw_dataset'] = {}
    recons_results['raw_dataset']['xvals'] = [float(x) for x in dataset.domain.meshgrid[0]]
    for i in range(dataset.ntrain):
        recons_results['raw_dataset']['train%g' % i] = {}
        recons_results['raw_dataset']['train%g' % i]['true_img'] = [float(d) for d in dataset.training_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['train%g' % i]['noisy_data'] = [float(d) for d in dataset.training_data[i][3].detach().numpy().flatten()]
    for i in range(dataset.ntest):
        recons_results['raw_dataset']['test%g' % i] = {}
        recons_results['raw_dataset']['test%g' % i]['true_img'] = [float(d) for d in dataset.test_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['test%g' % i]['noisy_data'] = [float(d) for d in dataset.test_data[i][3].detach().numpy().flatten()]

    # Save training data reconstructions
    print("Training data reconstruction")
    dataset, model, total_fista_iters, total_cg_iters, obj, obj_vec, quit_on_timeout \
        = calculate_upper_obj_and_gradient(dataset, model, fista_xtol=recons_fista_xtol, no_gradient=True, training_data=True,
                                           maxiters=None, verbose=True, verbose_to_print=True)
    recons_results['recons_train'] = {}
    recons_results['recons_train']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_train']['loss_mean'] = obj
    for i in range(dataset.ntrain):
        recons_results['recons_train']['train%g' % i] = [float(r) for r in np.array(dataset.training_recons[i]).flatten()]

    # Save test data reconstructions
    print("Test data reconstruction")
    dataset, model, total_fista_iters, total_cg_iters, obj, obj_vec, quit_on_timeout \
        = calculate_upper_obj_and_gradient(dataset, model, fista_xtol=recons_fista_xtol, no_gradient=True, training_data=False,
                                           maxiters=None, verbose=True, verbose_to_print=True)
    recons_results['recons_test'] = {}
    recons_results['recons_test']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_test']['loss_mean'] = obj
    for i in range(dataset.ntrain):
        recons_results['recons_test']['test%g' % i] = [float(r) for r in np.array(dataset.test_recons[i]).flatten()]

    return recons_results


def plot_ad_cg_bilevel_runs(font_size, fmt='eps'):
    # Plot results of bilevel runs using different AD methods/niters
    # Based on run_bilevel_ad_testing.py and plot_bilevel_ad_testing.py

    # Find all files with run information
    print("Loading run data")
    setting_name = 'denoising1d_pwlinear_shallow_final'
    this_results_folder = os.path.join('raw_results', setting_name)
    all_files = sorted(get_all_json_files(this_results_folder, must_contain='%s_ad_test_run' % setting_name))
    all_files = [os.path.join(this_results_folder, f) for f in all_files]

    # Load all run information
    all_results = {}  # key = (tol, ad_method_iters)
    for run_file in all_files:
        mydict = read_json(run_file)
        ad_method_iters = mydict['gd_solver']['ad_method_iters']
        tol = mydict['gd_solver']['init_tol']
        df = pd.read_csv(run_file.replace('.json', '.csv'))
        if len(df) == 0:
            print("Skipping empty csv file: %s" % run_file.replace('.json', '.csv'))
            continue
        df['total_fista_iters'] = df['fista_iters'].cumsum()
        df['total_cg_iters'] = df['cg_iters'].cumsum()
        df['total_fista_cg_iters'] = df['total_fista_iters'] + df['total_cg_iters']
        all_results[(tol, ad_method_iters)] = (mydict, df)
    all_tols = sorted(list(set([k[0] for k in all_results.keys()])))[::-1]
    all_ad_method_iters = sorted(list(set([k[1] for k in all_results.keys()])))[::-1]

    # Make relevant plots
    ad_method_iters_list = ['cg_10', 'cg_100', 'ad_gd_10', 'ad_gd_100', 'ad_hb_10', 'ad_hb_100']
    # all_tols = [0.01]
    print("Making plots (tol = %g)" % tol)
    for tol in all_tols:
        plot_single_ad_cg_bilevel_obj_dec(all_results, ad_method_iters_list, tol, font_size, by_runtime=False, fmt=fmt, figsize=cm2inch(6.1, 4.1))
        plot_single_ad_cg_bilevel_obj_dec(all_results, ad_method_iters_list, tol, font_size, by_runtime=True, fmt=fmt, figsize=cm2inch(6.1, 4.1))

        # For now, just create data with all reconstruction information
        for ad_method_iters in ad_method_iters_list:
            recons_results = save_single_ad_cg_bilevel_recons(all_results, ad_method_iters, tol, recons_fista_xtol=1e-5)
            save_dict(recons_results, os.path.join(OUTFOLDER, 'ad_cg_bilevel', '%s_tol%g_recons_data.json' % (ad_method_iters, tol)))
    return


def plot_tv_calibration(outfolder, font_size, get_best_weight_only=False, fmt='eps'):
    # Plot TV tuning results
    # Unless get_best_weight_only=True, then just return best value of alpha found

    setting_name = 'denoising1d_pwlinear_final_tv'
    this_results_folder = os.path.join('raw_results', setting_name)
    all_files = sorted(get_all_json_files(this_results_folder, must_contain='%s_init' % setting_name))
    all_files = [os.path.join(this_results_folder, f) for f in all_files]

    results = []
    for run_file in all_files:
        mydict = read_json(run_file)
        df = pd.DataFrame.from_dict(mydict["history"])
        cols_to_sum = [c for c in list(df.columns) if c.startswith('sqloss')]
        df['sqloss_avg'] = df[cols_to_sum].mean(axis=1)
        results.append(df)

    cols_to_merge = ['alpha', 'sqloss_avg']
    merged_results = pd.concat([d[cols_to_merge] for d in results], ignore_index=True)

    if get_best_weight_only:
        best_row = merged_results.loc[merged_results['sqloss_avg'].idxmin()]
        return float(best_row['alpha']), float(best_row['sqloss_avg'])
    else:
        # Do plotting
        plt.figure()
        plt.clf()
        ax = plt.gca()
        plt.loglog(merged_results['alpha'], merged_results['sqloss_avg'], 'b.', markersize=10)
        ax.set_xlabel(r"TV regularizer weight", fontsize=font_size)
        ax.set_ylabel("Average loss", fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        plt.grid()
        filename = 'tv_calibration.%s' % fmt
        plt.savefig(os.path.join(outfolder, filename), bbox_inches='tight')
        merged_results.to_csv(os.path.join(outfolder, filename.replace('.%s' % fmt, '.csv')), index=False)
        return


def save_tv_recons(best_alpha):
    setting_name = 'denoising1d_pwlinear_final_tv'
    setting_dict = read_json(os.path.join('settings_dfols', '%s.json' % setting_name))
    dataset = generate_dataset(setting_dict)
    pdhg_niters = int(setting_dict["regularizer"]["pdhg_niters"])
    print('niters = %g' % pdhg_niters)

    recons_results = {}
    recons_results['pdhg_niters'] = pdhg_niters
    recons_results['tv_alpha'] = best_alpha

    # Save raw data
    recons_results['raw_dataset'] = {}
    recons_results['raw_dataset']['xvals'] = [float(x) for x in dataset.domain.meshgrid[0]]
    for i in range(dataset.ntrain):
        recons_results['raw_dataset']['train%g' % i] = {}
        recons_results['raw_dataset']['train%g' % i]['true_img'] = [float(d) for d in dataset.training_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['train%g' % i]['noisy_data'] = [float(d) for d in dataset.training_data[i][3].detach().numpy().flatten()]
    for i in range(dataset.ntest):
        recons_results['raw_dataset']['test%g' % i] = {}
        recons_results['raw_dataset']['test%g' % i]['true_img'] = [float(d) for d in dataset.test_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['test%g' % i]['noisy_data'] = [float(d) for d in dataset.test_data[i][3].detach().numpy().flatten()]

    reg_fn = lambda noisy_data: tv_denoise(dataset.domain, dataset.odl_fwd_op, noisy_data,
                                           alpha=best_alpha, niters=pdhg_niters, use_custom_pdhg=False)

    # Training reconstruction
    obj_vec = np.zeros((dataset.ntrain,), dtype=float)
    for i in range(dataset.ntrain):
        print(" - Train %g/%g" % (i + 1, dataset.ntrain))
        recons_odl = reg_fn(dataset.get_noisy_img(i, training_data=True, as_tensor=False))
        dataset.training_recons[i] = numpy_to_tensor(recons_odl.asarray())  # save reconstruction
        obj_vec[i] = float(dataset.upper_objfun(i, training_data=True))

    recons_results['recons_train'] = {}
    recons_results['recons_train']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_train']['loss_mean'] = float(np.mean(obj_vec))
    for i in range(dataset.ntrain):
        recons_results['recons_train']['train%g' % i] = [float(r) for r in np.array(dataset.training_recons[i]).flatten()]

    # Test reconstruction
    obj_vec = np.zeros((dataset.ntest,), dtype=float)
    for i in range(dataset.ntest):
        print(" - Test %g/%g" % (i + 1, dataset.ntest))
        recons_odl = reg_fn(dataset.get_noisy_img(i, training_data=False, as_tensor=False))
        dataset.test_recons[i] = numpy_to_tensor(recons_odl.asarray())  # save reconstruction
        obj_vec[i] = float(dataset.upper_objfun(i, training_data=False))

    recons_results['recons_test'] = {}
    recons_results['recons_test']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_test']['loss_mean'] = float(np.mean(obj_vec))
    for i in range(dataset.ntrain):
        recons_results['recons_test']['test%g' % i] = [float(r) for r in np.array(dataset.test_recons[i]).flatten()]

    return recons_results


def plot_tv_results(font_size, fmt='eps'):
    # Plot results from standalone TV calibration
    outfolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison')

    plot_tv_calibration(outfolder, font_size, fmt=fmt)  # do plotting

    best_alpha, best_loss = plot_tv_calibration(outfolder, font_size, get_best_weight_only=True, fmt=fmt)  # get alpha
    print("Doing TV reconstructions with best alpha = %g" % best_alpha)
    tv_recons = save_tv_recons(best_alpha)
    save_dict(tv_recons, os.path.join(outfolder, 'tv_recons_data.json'))
    return


def plot_bilevel_decrease(results, stepsize_to_use, tols, font_size, by_runtime=False, fmt='eps', figsize=None):
    outfolder = os.path.join(OUTFOLDER, 'bilevel_tv_comparison')

    plt.figure(figsize=figsize)
    plt.clf()
    ax = plt.gca()
    raw_results = {}
    for tol in tols:
        if (tol, stepsize_to_use) in results:
            mydict, df = results[(tol, stepsize_to_use)]
            xvals = df['wall_runtime'].to_numpy() if by_runtime else df['total_fista_cg_iters'].to_numpy()
            yvals = df['obj'].to_numpy()
            # if obj_best_so_far:
            #     yvals = np.minimum.accumulate(yvals)
            # Plot, with consistent formatting across different methods/iters
            lbl = 'Tol = %g' % tol
            plot_fun = plt.semilogy if by_runtime else plt.loglog  # use loglog for FISTA/AD iters plot
            plot_fun(xvals, yvals, linewidth=2, label=lbl)
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


def save_single_bilevel_tv_comparison_recons(results, stepsize, tol, recons_fista_xtol=1e-5):
    best_params = get_best_params(results)
    mydict, df = results[(tol, stepsize)]

    model, dataset = get_model_and_dataset(mydict)
    p = model.parameter_values()
    pnew = torch.tensor(best_params[(tol, stepsize)]).reshape(p.shape)
    model.set_params(pnew)

    recons_results = {}
    recons_results['tol'] = tol
    recons_results['stepsize'] = stepsize
    recons_results['recons_fista_xtol'] = recons_fista_xtol
    recons_results['model_params'] = [float(p) for p in best_params[(tol, stepsize)].flatten()]

    # Save raw data
    recons_results['raw_dataset'] = {}
    recons_results['raw_dataset']['xvals'] = [float(x) for x in dataset.domain.meshgrid[0]]
    for i in range(dataset.ntrain):
        recons_results['raw_dataset']['train%g' % i] = {}
        recons_results['raw_dataset']['train%g' % i]['true_img'] = [float(d) for d in dataset.training_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['train%g' % i]['noisy_data'] = [float(d) for d in dataset.training_data[i][3].detach().numpy().flatten()]
    for i in range(dataset.ntest):
        recons_results['raw_dataset']['test%g' % i] = {}
        recons_results['raw_dataset']['test%g' % i]['true_img'] = [float(d) for d in dataset.test_data[i][2].detach().numpy().flatten()]
        recons_results['raw_dataset']['test%g' % i]['noisy_data'] = [float(d) for d in dataset.test_data[i][3].detach().numpy().flatten()]

    # Save training data reconstructions
    print("Training data reconstruction")
    dataset, model, total_fista_iters, total_cg_iters, obj, obj_vec, quit_on_timeout \
        = calculate_upper_obj_and_gradient(dataset, model, fista_xtol=recons_fista_xtol, no_gradient=True, training_data=True,
                                           maxiters=None, verbose=True, verbose_to_print=True)
    recons_results['recons_train'] = {}
    recons_results['recons_train']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_train']['loss_mean'] = obj
    for i in range(dataset.ntrain):
        recons_results['recons_train']['train%g' % i] = [float(r) for r in np.array(dataset.training_recons[i]).flatten()]

    # Save test data reconstructions
    print("Test data reconstruction")
    dataset, model, total_fista_iters, total_cg_iters, obj, obj_vec, quit_on_timeout \
        = calculate_upper_obj_and_gradient(dataset, model, fista_xtol=recons_fista_xtol, no_gradient=True, training_data=False,
                                           maxiters=None, verbose=True, verbose_to_print=True)
    recons_results['recons_test'] = {}
    recons_results['recons_test']['loss_vec'] = [float(v) for v in obj_vec]
    recons_results['recons_test']['loss_mean'] = obj
    for i in range(dataset.ntrain):
        recons_results['recons_test']['test%g' % i] = [float(r) for r in np.array(dataset.test_recons[i]).flatten()]

    return recons_results


def plot_bilevel_tv_comparison(font_size, fmt='eps'):
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
    tols = [0.01, 0.001, 0.0001]  # just show the ones with very long runtimes
    for stepsize in stepsizes:
        plot_bilevel_decrease(results, stepsize, tols, font_size, by_runtime=False, fmt=fmt, figsize=cm2inch(8.1, 4.1))
        plot_bilevel_decrease(results, stepsize, tols, font_size, by_runtime=True, fmt=fmt, figsize=cm2inch(8.1, 4.1))
        for tol in tols:
            print("Saving reconstruction data for stepsize = %g, tol = %g" % (stepsize, tol))
            recons_results = save_single_bilevel_tv_comparison_recons(results, stepsize, tol, recons_fista_xtol=1e-5)
            save_dict(recons_results, os.path.join(outfolder, 'bilevel_stepsize%g_tol%g_recons_data.json' % (stepsize, tol)))
    return


def main():
    # font_size = 'large'  # x-large for presentations
    font_size = 'x-small'  # x-large for presentations
    fmt = 'pdf'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # plot_ad_cg_testing(font_size, fmt=fmt)
    # plot_ad_cg_bilevel_runs(font_size, fmt=fmt)
    # plot_tv_results(font_size, fmt=fmt)

    # TODO when have long duration bilevel_fixed_restarts.py results, rerun the below (check fmt='pdf' above)
    # TODO then rerun plot_recons.py
    plot_bilevel_tv_comparison(font_size, fmt=fmt)
    print("Done")
    return


if __name__ == '__main__':
    main()
