#!/usr/bin/env python3

"""
Routines for running inexact GD on the upper-level learning problem

Based on old_scripts/upper_solver.py
"""
from datetime import datetime, timedelta
import logging
from math import sqrt
import numpy as np
import os
import pandas as pd
import torch

from tune_standard_regularizers import generate_dataset
from nn_regularizers import ShallowConv1d, VDeepConv1d
try:
    from ad_testing_v2.utils import AbstractDataset, DenoisingDataset1D, DenoisingDataset2D, SuperresolutionDataset2D
    from ad_testing_v2.utils import estimate_fista_maxiters, fista, conjugate_gradient, BaseConvexRegularizer
    from ad_testing_v2.utils import read_json, save_dict
except ModuleNotFoundError:
    from utils import AbstractDataset, DenoisingDataset1D, DenoisingDataset2D, SuperresolutionDataset2D
    from utils import estimate_fista_maxiters, fista, conjugate_gradient, BaseConvexRegularizer
    from utils import read_json, save_dict


def inexact_ad(gradf, hess_vec_prod, jac_vec_prod, g0, step_size, heavy_ball_momentum, niters, save_history=False):
    copy = lambda v: v.detach().clone()  # make copy of a tensor

    xbar_inexact = copy(gradf)
    xbar_inexact_prev = torch.zeros_like(xbar_inexact)
    upper_grad = g0
    if save_history:
        ghist = [copy(upper_grad)]
    else:
        ghist = None

    for i in range(niters):
        upper_grad -= step_size * jac_vec_prod(xbar_inexact)
        xbar_inexact_new = (1.0 + heavy_ball_momentum) * xbar_inexact - step_size * hess_vec_prod(xbar_inexact) - heavy_ball_momentum * xbar_inexact_prev
        xbar_inexact_prev = copy(xbar_inexact)
        xbar_inexact = xbar_inexact_new
        if save_history:
            ghist.append(copy(upper_grad))

    if save_history:
        return upper_grad, niters, ghist
    else:
        return upper_grad, niters


def calculate_upper_obj_and_gradient(dataset: AbstractDataset, model: BaseConvexRegularizer, fista_xtol=1e-5,
                                     cg_rtol=1e-5, ad_method_iters=None, verbose=False, no_gradient=False, verbose_to_print=False,
                                     training_data=True, maxiters=None, fista_verbose=False,
                                     max_wall_runtime_seconds=None, tic=None):
    """
    Evaluate upper level objective and gradient to within some tolerances:
    - fista_xtol determines accuracy of lower-level reconstructions, ||Xk-X*|| <= fista_xtol
    - cg_rtol determines accuracy of upper-level gradient linear solver
    - Alternative to cg_rtol is ad_method_iters = 'cg_100', 'ad_hb_100', etc. for method + niters

    Overall gradient accuracy is ||inexact_grad - true_grad|| = O(fista_xtol + cg_rtol)
    """
    if cg_rtol is not None:
        assert ad_method_iters is None, "Cannot specify ad_method_iters if cg_rtol is set"
    else:
        assert ad_method_iters is not None, "Must specify ad_method_iters if cg_rtol=None"
    # Check for timeout
    timeout_exceeded = lambda toc: (toc - tic).total_seconds() > max_wall_runtime_seconds if max_wall_runtime_seconds is not None else False

    # Logging helper function
    def log(msg):
        if verbose:
            if verbose_to_print:
                print(msg)
            else:
                logging.info(msg)
        return

    total_obj = 0.0
    total_grad = None
    total_fista_iters = 0
    total_cg_iters = 0
    nimgs = dataset.ntrain if training_data else dataset.ntest
    obj_vec = np.zeros((nimgs,))
    log("Starting evaluation:")
    for i in range(nimgs):
        model.current_index = i
        model.current_recons_training_data = training_data
        recons = dataset.training_recons[i] if training_data else dataset.test_recons[i]
        estimated_maxiters = estimate_fista_maxiters(model, recons, ytol=fista_xtol)
        log("%g: Expecting max %g FISTA iters" % (i, estimated_maxiters))
        img_recons, obj, gnorm, niters, terminated_on_small_gradient = fista(model, recons, ytol=fista_xtol,
                                                                             niters=None if maxiters is None else maxiters - total_fista_iters - total_cg_iters,
                                                                             verbose=fista_verbose,
                                                                             max_wall_runtime_seconds=max_wall_runtime_seconds,
                                                                             tic=tic)
        total_fista_iters += niters
        log("%g: FISTA finished after %g iters" % (i, niters))
        if training_data:
            dataset.training_recons[i] = img_recons
        else:
            dataset.test_recons[i] = img_recons

        if timeout_exceeded(datetime.now()):
            log("Timeout exceeded, quitting calculate_upper_obj_and_gradient")
            break

        # Calculate this component of upper-level objective and gradient
        upper_obj = float(dataset.upper_objfun(i, training_data=training_data))
        total_obj += upper_obj
        obj_vec[i] = upper_obj
        if not no_gradient:
            gradf = dataset.upper_gradient_wrt_img_recons(i, training_data=training_data)
            Hv = lambda v: model.hess_vec_wrt_inputs(img_recons, v)
            Jv = lambda v: model.jac_vec(img_recons, v)
            if not torch.isnan(gradf).any():
                if cg_rtol is not None:
                    # For main runs: use CG with relative error bound
                    q, num_cg_iters, _ = conjugate_gradient(Hv, gradf, torch.ones_like(gradf), rtol=cg_rtol, maxiter=None, verbose=False)
                    upper_grad = -Jv(q)
                else:
                    # AD testing runs - CG/AD with fixed number of iterations
                    g0 = torch.zeros_like(model.parameter_values())  # starting guess for fixed AD iterations
                    L, mu = model.lip_const(), model.convex_const()  # used for AD parameters
                    if ad_method_iters.startswith('cg'):
                        num_cg_iters = int(ad_method_iters.replace('cg_', ''))
                        log("%g: getting gradient with %g CG iters" % (i, num_cg_iters))
                        # Normally we would just run until niters, but if niters is large then CG can produce some nan
                        # values because the residual gets so small. So, save the whole history (up to length niters)
                        # and take the last value which does not have nans
                        _, _, _, qhist = conjugate_gradient(Hv, gradf, torch.zeros_like(gradf), rtol=1e-15,
                                                            maxiter=num_cg_iters, verbose=False, save_history=True)
                        upper_grad_hist = [-Jv(q) for q in qhist]
                        while torch.isnan(upper_grad_hist[-1]).any():
                            upper_grad_hist.pop(-1)
                        upper_grad = upper_grad_hist[-1]
                    elif ad_method_iters.startswith('ad_gd'):
                        num_cg_iters = int(ad_method_iters.replace('ad_gd_', ''))
                        log("%g: getting gradient with %g AD/GD iters" % (i, num_cg_iters))
                        step_size = 2.0 / (L + mu)
                        # step_size = 1.0 / L
                        heavy_ball_momentum = 0.0
                        upper_grad, _ = inexact_ad(gradf, Hv, Jv, g0, step_size, heavy_ball_momentum, num_cg_iters,
                                                   save_history=False)
                    elif ad_method_iters.startswith('ad_hb'):
                        num_cg_iters = int(ad_method_iters.replace('ad_hb_', ''))
                        log("%g: getting gradient with %g AD/HB iters" % (i, num_cg_iters))
                        qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
                        heavy_ball_momentum = qHB ** 2  # beta
                        step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha
                        # step_size = 1.0 * (1.0 + heavy_ball_momentum) / L  # alpha
                        upper_grad, _ = inexact_ad(gradf, Hv, Jv, g0, step_size, heavy_ball_momentum, num_cg_iters,
                                                   save_history=False)
                    else:
                        raise RuntimeError("Unknown ad_method_iters: %s" % ad_method_iters)
                total_cg_iters += num_cg_iters
            else:
                upper_grad = float('nan') * torch.ones_like(gradf)
            log("%g: %g, %s" % (i, obj, str(upper_grad)))
            if total_grad is None:
                total_grad = upper_grad
            else:
                total_grad += upper_grad
        else:
            log("%g: %g" % (i, obj))

        if timeout_exceeded(datetime.now()):
            log("Timeout exceeded, quitting calculate_upper_obj_and_gradient")
            break

    quit_on_timeout = timeout_exceeded(datetime.now())
    # Convert to mean obj and grad
    if no_gradient:
        return dataset, model, total_fista_iters, total_cg_iters, total_obj / nimgs, obj_vec, quit_on_timeout
    else:
        return dataset, model, total_fista_iters, total_cg_iters, total_obj / nimgs, obj_vec, total_grad / nimgs, quit_on_timeout


def inexact_gd(dataset: AbstractDataset, model: BaseConvexRegularizer, max_lower_iters: int,
               init_tol=1.0, init_stepsize=1.0, decreasing_tol_stepsize=False, verbose=False, fista_verbose=False,
               decrease_tol_stepsize_timestep_in_hours=None,
               old_niters=None, old_runtime=None, ad_method_iters=None,
               verbose_to_print=False, save_params=True, max_flat_iters=100, flat_thresh=1e-5, save_train_recons=False,
               max_wall_runtime_seconds=None) -> pd.DataFrame:
    """
    Solve the upper-level problem with inexact projected GD and some set of tolerances/stepsizes (fixed or decreasing).

    Run for a total amount of computational work (measured as total FISTA+CG iterations across all training samples)

    Uses init_tol and init_stepsize as tolerance and stepsizes throughout, unless decreasing_tol_stepsize=True.
    In that case, decrease init_tol/stepsize by a factor of 10 every N hours of runtime.
    """
    if decreasing_tol_stepsize:
        assert decrease_tol_stepsize_timestep_in_hours is not None, "Need to set decrease_tol_stepsize_timestep_in_hours when running with decreasing_tol_stepsize"
    else:
        assert decrease_tol_stepsize_timestep_in_hours is None, "Cannot set decrease_tol_stepsize_timestep_in_hours for fixed accuracy runs"

    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    # Logging helper function
    def log(msg):
        if verbose:
            if verbose_to_print:
                print(msg)
            else:
                logging.info(msg)
        return

    # Save data
    run_data = {}
    for col in ['k', 'obj_grad_tol', 'alpha', 'obj', 'normg', 'fista_iters', 'cg_iters', 'wall_runtime']:
        run_data[col] = []
    for i in range(dataset.ntrain):
        run_data['obj_vec%g' % i] = []
    if save_params:
        for i in range(model.numel()):
            run_data['param%g' % i] = []
    if save_train_recons:
        for i in range(dataset.ntrain):
            this_data = dataset.training_recons[i].detach().numpy().flatten()
            for j in range(len(this_data)):
                run_data['train%g_%g' % (i, j)] = []

    k = 0 if old_niters is None else old_niters + 1
    total_lower_iters = 0
    if max_flat_iters is not None:
        obj_history = np.nan * np.ones((max_flat_iters,))
        num_iters_no_lower_work = 0
    else:
        obj_history = None
        num_iters_no_lower_work = None

    # Allow termination on max runtime
    tic = datetime.now()
    if old_runtime is not None:
        tic = tic - timedelta(seconds=old_runtime)
    timeout_exceeded = lambda toc: (toc-tic).total_seconds() > max_wall_runtime_seconds if max_wall_runtime_seconds is not None else False
    while total_lower_iters < max_lower_iters and not timeout_exceeded(datetime.now()):
        # Evaluate objective + gradient at current parameters
        log("**** k = %g ****" % k)
        if decreasing_tol_stepsize:
            runtime_in_hours = (datetime.now() - tic).total_seconds() / 3600.0
            n_decreases = int(runtime_in_hours // decrease_tol_stepsize_timestep_in_hours)
            obj_grad_tol = init_tol * (0.1 ** n_decreases)
            alpha = init_stepsize * (0.1 ** n_decreases)
        else:
            obj_grad_tol = init_tol
            alpha = init_stepsize

        if ad_method_iters is not None:
            log("Using obj_tol = %g and alpha = %g and gradient method %s" % (obj_grad_tol, alpha, ad_method_iters))
        else:
            log("Using obj_grad_tol = %g and alpha = %g" % (obj_grad_tol, alpha))

        params = model.parameter_values()
        try:
            dataset, model, total_fista_iters, total_cg_iters, obj, obj_vec, g, quit_on_timeout \
                = calculate_upper_obj_and_gradient(dataset, model,
                                                   fista_xtol=obj_grad_tol,
                                                   cg_rtol=(obj_grad_tol if ad_method_iters is None else None),
                                                   ad_method_iters=ad_method_iters,
                                                   verbose=verbose, verbose_to_print=verbose_to_print, training_data=True,
                                                   maxiters=max_lower_iters - total_lower_iters, fista_verbose=fista_verbose,
                                                   max_wall_runtime_seconds=max_wall_runtime_seconds, tic=tic)
        except:
            log("Objective/gradient evaluation failed, terminating")
            break

        if quit_on_timeout:
            break

        log("Here, obj = %g and ||grad||= %g" % (obj, norm(g)))
        total_lower_iters += total_fista_iters
        total_lower_iters += total_cg_iters

        # Save obj history
        if obj_history is not None:
            if k < max_flat_iters:
                obj_history[k] = obj
            else:
                # Move everything forward, so we have: obj_history = [f(k-N+1), ..., f(k)]
                obj_history[:-1] = obj_history[1:]
                obj_history[-1] = obj

        if num_iters_no_lower_work is not None:
            if total_fista_iters == dataset.ntrain and total_cg_iters == dataset.ntrain:
                num_iters_no_lower_work += 1
            else:
                num_iters_no_lower_work = 0

        # Store data from this iteration
        run_data['k'].append(k)
        run_data['obj_grad_tol'].append(obj_grad_tol)
        run_data['alpha'].append(alpha)
        run_data['obj'].append(obj)
        run_data['normg'].append(norm(g))
        run_data['fista_iters'].append(total_fista_iters)
        run_data['cg_iters'].append(total_cg_iters)
        run_data['wall_runtime'].append((datetime.now()-tic).total_seconds())
        for i in range(dataset.ntrain):
            run_data['obj_vec%g' % i].append(obj_vec[i])
        if save_params:
            p = model.parameter_values(as_numpy=True)
            for i in range(model.numel()):
                run_data['param%g' % i].append(p[i])
        if save_train_recons:
            for i in range(dataset.ntrain):
                this_data = dataset.training_recons[i].detach().numpy().flatten()
                for j in range(len(this_data)):
                    run_data['train%g_%g' % (i, j)] = this_data[j]

        try:
            params_new = params - alpha * g
        except:
            log("Terminating as params update failed")
            break

        if torch.isnan(params_new).any():
            log("Terminating because of NaNs")
            break

        model.set_params(params_new)
        model.project_weights_to_feasible_set()

        log("After iter %g" % k)
        log("Params = %s" % str(model.parameter_values(as_numpy=True)))

        # Terminate if obj decreasing too slowly or have too many iters with no FISTA/CG work
        if obj_history is not None and (k > max_flat_iters or num_iters_no_lower_work > max_flat_iters):
            max_obj = np.max(obj_history)
            min_obj = np.min(obj_history)
            rel_chg = (max_obj - min_obj) / max_obj
            if rel_chg < flat_thresh:
                log("Terminating on slow objective decrease")
                break

        # Plot results (not used)
        # if k % save_plot_freq == 0:
        #     dataset.plot(title="Iter %g, obj = %g" % (k, obj))
        #     if save_plots:
        #         plt.savefig('img/%s/gd%g.png' % (run_name, k), bbox_inches='tight')
        #     else:
        #         plt.show()

        k += 1

    if timeout_exceeded(datetime.now()):
        log("Exceeded maximum runtime of %g seconds" % max_wall_runtime_seconds)

    # Clean up
    log("Run finished")
    df = pd.DataFrame.from_dict(run_data)
    return df


def get_model_and_dataset(settings_dict: dict) -> (BaseConvexRegularizer, AbstractDataset):
    # Already built the dataset generation function in tune_standard_regularizers.py
    dataset = generate_dataset(settings_dict)

    # Read model settings
    model_settings = settings_dict["regularizer"]
    model_type = str(model_settings["type"])
    beta = float(model_settings["beta"])
    conv_start_value = float(model_settings["conv_start_value"])
    L2_start_value = float(model_settings["L2_start_value"])
    have_fixed_L2 = bool(model_settings["have_fixed_L2"])
    init_weights_min = float(model_settings["init_weights_min"])
    init_weights_max = float(model_settings["init_weights_max"])
    seed = int(model_settings["seed"])

    # Create model
    if model_type == 'shallow1d':
        symmetric_kernel = bool(model_settings["symmetric_kernel"])
        model = ShallowConv1d(dataset, beta=beta, conv_start_value=conv_start_value, L2_start_value=L2_start_value,
                              have_fixed_L2=have_fixed_L2, symmetric_kernel=symmetric_kernel, keep_tv_kernel=False)
    elif model_type == 'vdeep1d':
        model = VDeepConv1d(dataset, beta=beta, conv_start_value=conv_start_value, L2_start_value=L2_start_value,
                              have_fixed_L2=have_fixed_L2)
    else:
        raise RuntimeError("Unknown model_type: '%s'" % model_type)

    # Weight initialization
    model.initialize_weights_random(min_val=init_weights_min, max_val=init_weights_max)  # initial arbitrary init (not used)
    np.random.seed(seed)
    init_wts = model.parameter_values()
    if "init_params" in model_settings:
        if model_settings["init_params"] == "tv_shallow1d":
            # Initialize shallow 1d network with TV
            model.initialize_weights_tv()
            model_init_wts = model.parameter_values()
        elif model_settings["init_params"] == "tv_vdeep1d":
            # Initialize vdeep 1d network with TV
            model.initialize_weights_tv()
            model_init_wts = model.parameter_values()
        elif model_settings["init_params"] == "good_pwlinear_shallow1d":
            # Initialize shallow 1d network with good results
            # Taken from end point of step 0.1, tol 0.1 [run 7] of denoising1d_pwlinear_shallow
            infile = os.path.join('raw_results', 'denoising1d_pwlinear_shallow', 'denoising1d_pwlinear_shallow_run7.csv')
            df = pd.read_csv(infile)
            param_cols = [c for c in list(df.columns) if c.startswith('param')]
            final_params = df[param_cols].iloc[-1].to_numpy()
            model_init_wts = torch.Tensor(final_params)
        elif model_settings["init_params"] == "good_pwlinear_vdeep1d":
            # Initialize vdeep 1d network with good results
            # Taken from step 1, tol 1 [run 0] of denoising1d_pwlinear_deep
            infile = os.path.join('raw_results', 'denoising1d_pwlinear_deep', 'denoising1d_pwlinear_deep_run0.csv')
            df = pd.read_csv(infile)
            param_cols = [c for c in list(df.columns) if c.startswith('param')]
            final_params = df[param_cols].iloc[-1].to_numpy()
            model_init_wts = torch.Tensor(final_params)
        else:
            raise RuntimeError("Unknown model setting 'init_params': %s" % model_settings["init_params"])
        model_init_wts = torch.reshape(model_init_wts, init_wts.shape)
    else:
        model_init_wts = torch.from_numpy(np.random.uniform(low=init_weights_min, high=init_weights_max, size=init_wts.shape))
    model.set_params(model_init_wts)

    return model, dataset


def get_final_params(df):
    num_param_cols = len([c for c in list(df.columns) if c.startswith('param')])
    param_cols = ['param%g' % i for i in range(num_param_cols)]
    return df[param_cols].iloc[-1].to_numpy()


def get_final_recons(df, ntrain):
    recons = []
    for i in range(ntrain):
        num_param_cols = len([c for c in list(df.columns) if c.startswith('train%g_' % i)])
        param_cols = ['train%g_%g' % (i, j) for j in range(num_param_cols)]
        recons.append(df[param_cols].iloc[-1].to_numpy())
    return recons


def do_single_solve(settings_dict: dict, max_lower_iters_per_image: int, init_tol: float, init_stepsize: float,
                    decreasing_tol_stepsize: bool, max_flat_iters: int, flat_thresh: float, save_train_recons=False,
                    decrease_tol_stepsize_timestep_in_hours=None, previous_run_csv_file=None, ad_method_iters=None,
                    verbose=False, verbose_to_print=False, max_wall_runtime_seconds=None) -> (dict, pd.DataFrame):
    model, dataset = get_model_and_dataset(settings_dict)
    max_lower_iters = max_lower_iters_per_image * dataset.ntrain

    if previous_run_csv_file is not None:
        assert '1d' in settings_dict['dataset']['type'], "Can only load saved 1d data so far"
        # Start run from end point of previous run
        old_df = pd.read_csv(previous_run_csv_file)

        # Set model parameters
        old_final_params = get_final_params(old_df)
        p = model.parameter_values()
        pnew = torch.tensor(old_final_params).reshape(p.shape)
        model.set_params(pnew)

        # Update dataset reconstructions TODO this only works in 1d
        old_final_recons = get_final_recons(old_df, dataset.ntrain)
        for i in range(dataset.ntrain):
            old_recons = old_final_recons[i]
            dataset.training_recons[i][0, 0, :] = torch.tensor(old_recons)

        old_niters = int(old_df['k'].iloc[-1])
        old_runtime = float(old_df['wall_runtime'].iloc[-1])
    else:
        old_df = None
        old_niters = None
        old_runtime = None

    output_settings_dict = settings_dict.copy()
    output_settings_dict["gd_solver"] = {}
    output_settings_dict["gd_solver"]["max_lower_iters_per_image"] = max_lower_iters_per_image
    output_settings_dict["gd_solver"]["max_lower_iters"] = max_lower_iters
    output_settings_dict["gd_solver"]["init_tol"] = init_tol
    output_settings_dict["gd_solver"]["init_stepsize"] = init_stepsize
    output_settings_dict["gd_solver"]["decreasing_tol_stepsize"] = decreasing_tol_stepsize
    if decreasing_tol_stepsize:
        output_settings_dict["gd_solver"]["decrease_tol_stepsize_timestep_in_hours"] = decrease_tol_stepsize_timestep_in_hours
    if ad_method_iters is not None:
        output_settings_dict["gd_solver"]["ad_method_iters"] = ad_method_iters
    output_settings_dict["gd_solver"]["max_flat_iters"] = max_flat_iters
    output_settings_dict["gd_solver"]["flat_thresh"] = flat_thresh
    output_settings_dict["gd_solver"]["save_train_recons"] = save_train_recons
    if max_wall_runtime_seconds is not None:
        output_settings_dict["gd_solver"]["max_wall_runtime_seconds"] = max_wall_runtime_seconds

    dataset.reset_reconstructions()
    tic = datetime.now()
    if old_runtime is not None:
        tic = tic - timedelta(seconds=old_runtime)

    df = inexact_gd(dataset, model, max_lower_iters, init_tol=init_tol, init_stepsize=init_stepsize,
                    decreasing_tol_stepsize=decreasing_tol_stepsize, max_flat_iters=max_flat_iters,
                    old_niters=old_niters, old_runtime=old_runtime, ad_method_iters=ad_method_iters,
                    decrease_tol_stepsize_timestep_in_hours=decrease_tol_stepsize_timestep_in_hours,
                    flat_thresh=flat_thresh, verbose=verbose, verbose_to_print=verbose_to_print,
                    max_wall_runtime_seconds=max_wall_runtime_seconds, save_train_recons=save_train_recons)
    toc = datetime.now()
    output_settings_dict['time_start'] = tic.strftime("%Y-%m-%d %H:%M:%S")
    output_settings_dict['time_end'] = toc.strftime("%Y-%m-%d %H:%M:%S")
    output_settings_dict['runtime_wall'] = (toc - tic).total_seconds()

    if old_df is not None:
        df = pd.concat([old_df, df], ignore_index=True)

    return output_settings_dict, df


def do_single_run(settings_file: str, max_lower_iters_per_image: int, idx: int, init_tol: float, init_stepsize: float,
                  decreasing_tol_stepsize: bool, max_flat_iters: int, flat_thresh: float, save_train_recons=False,
                  decrease_tol_stepsize_timestep_in_hours=None, previous_run_csv_file=None, ad_method_iters=None,
                  settings_folder='settings_gd', output_folder='raw_results', verbose=False, verbose_to_print=False,
                  max_wall_runtime_seconds=None):
    base_run_name = settings_file.replace('.json', '')
    if decreasing_tol_stepsize:
        assert ad_method_iters is None, "Must use relative accuracy for decreasing_tol_stepsize runs"
        run_stem = '%s_dynamic_run%g' % (base_run_name, idx)
    elif ad_method_iters is not None:
        run_stem = '%s_ad_test_run%g' % (base_run_name, idx)
    else:
        run_stem = '%s_run%g' % (base_run_name, idx)
    settings_dict = read_json(os.path.join(settings_folder, settings_file))

    output_folder = os.path.join(output_folder, base_run_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if decreasing_tol_stepsize:
        print("Starting dynamic (%g hrs) run %s for init_tol %g and init_stepsize %g" % (decrease_tol_stepsize_timestep_in_hours, base_run_name, init_tol, init_stepsize))
        logging.info("Starting dynamic (%g hrs) run %s for init_tol %g and init_stepsize %g" % (decrease_tol_stepsize_timestep_in_hours, base_run_name, init_tol, init_stepsize))
    elif ad_method_iters is not None:
        print("Starting AD test %s run %s for init_tol %g and init_stepsize %g" % (ad_method_iters, base_run_name, init_tol, init_stepsize))
        logging.info("Starting AD test %s run %s for init_tol %g and init_stepsize %g" % (ad_method_iters, base_run_name, init_tol, init_stepsize))
    else:
        print("Starting run %s for init_tol %g and init_stepsize %g" % (base_run_name, init_tol, init_stepsize))
        logging.info("Starting run %s for init_tol %g and init_stepsize %g" % (base_run_name, init_tol, init_stepsize))
    output_info, df = do_single_solve(settings_dict, max_lower_iters_per_image, init_tol, init_stepsize,
                                      decreasing_tol_stepsize, max_flat_iters, flat_thresh,
                                      decrease_tol_stepsize_timestep_in_hours=decrease_tol_stepsize_timestep_in_hours,
                                      previous_run_csv_file=previous_run_csv_file,
                                      ad_method_iters=ad_method_iters,
                                      verbose=verbose, verbose_to_print=verbose_to_print,
                                      max_wall_runtime_seconds=max_wall_runtime_seconds,
                                      save_train_recons=save_train_recons)
    save_dict(output_info, os.path.join(output_folder, '%s.json' % run_stem))
    df.to_csv(os.path.join(output_folder, '%s.csv' % run_stem), index=False)
    return


def do_batch_solve(settings_file: str, max_lower_iters_per_image: int, init_tol_list: list, init_stepsize_list: list,
                   decreasing_tol_stepsize: bool, max_flat_iters: int, flat_thresh: float, save_train_recons=False,
                   decrease_tol_stepsize_timestep_in_hours=None,
                   settings_folder='settings_gd', output_folder='raw_results', verbose=False, verbose_to_print=False):
    idx = 0
    for init_tol in init_tol_list:
        for init_stepsize in init_stepsize_list:
            # Do this run and save results
            do_single_run(settings_file, max_lower_iters_per_image, idx, init_tol, init_stepsize,
                          decreasing_tol_stepsize, max_flat_iters, flat_thresh, settings_folder=settings_folder,
                          decrease_tol_stepsize_timestep_in_hours=decrease_tol_stepsize_timestep_in_hours,
                          output_folder=output_folder, verbose=verbose, verbose_to_print=verbose_to_print,
                          save_train_recons=save_train_recons)
            idx += 1
    return


def main():
    # Example run
    settings_folder = 'settings_gd'
    settings_file = 'denoising1d_rect_shallow.json'
    settings_dict = read_json(os.path.join(settings_folder, settings_file))

    max_lower_iters = 1000
    init_tol = 1e-1
    init_stepsize = 1e-2
    decreasing_tol_stepsize = False
    output_info, df = do_single_solve(settings_dict, max_lower_iters, init_tol, init_stepsize, decreasing_tol_stepsize)

    # Save results somewhere for inspection
    base_run_name = settings_file.replace('.json', '')
    output_folder = os.path.join('raw_results', base_run_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    save_dict(output_info, os.path.join(output_folder, '%s_test.json' % base_run_name))
    df.to_csv(os.path.join(output_folder, '%s_test.csv' % base_run_name), index=False)
    return


if __name__ == '__main__':
    main()
