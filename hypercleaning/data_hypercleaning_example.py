#!/usr/bin/env python3

"""
Compare AD methods on data hypercleaning problem from mnist_exp.py in
    https://github.com/JunjieYang97/MRVRBO
This problem/implementation is described fully in [1]

[1] J. Yang, K. Ji, and Y. Liang. Provably Faster Algorithms for Bilevel Optimization, NeurIPS (2021), http://arxiv.org/abs/2106.04692
"""
import argparse
from datetime import datetime
from math import sqrt, ceil, log
import os
import pandas as pd
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # Settings that I don't expect to be changed between runs
    parser.add_argument('--data_path', type=str, default='data/', help='The temporary data storage path')
    parser.add_argument('--results_path', type=str, default='raw_hypercleaning_results/', help='Folder to store raw results')
    parser.add_argument('--noise_rate', type=float, default=0.1, help='Fraction of corrupted training labels')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes/labels')
    parser.add_argument('--training_size', type=int, default=20000, help='Number of lower-level samples')
    parser.add_argument('--validation_size', type=int, default=5000, help='Number of upper-level samples')
    parser.add_argument('--reg_value', type=float, default=0.001, help='L2 regularizer weight')
    parser.add_argument('--device', type=str, default='cpu', help='Pytorch device for data')
    parser.add_argument('--seed', type=int, default=0, help='Pytorch starting random seed')
    parser.add_argument('--total_budget', type=int, default=100000, help='Total lower-level + AD iterations budget')
    
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print lots of information')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false', help='Print limited information')
    parser.set_defaults(verbose=False)
    
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save results to file')
    parser.add_argument('--no_save_results', dest='save_results', action='store_false', help='Don\'t  results to file')
    parser.set_defaults(save_results=True)

    # Settings that I expect to be changed between runs
    parser.add_argument('--lower_tol', type=float, default=0.01, help='Tolerance ||Y-Y*|| <= eps for lower-level solver')
    parser.add_argument('--lower_solver', type=str, default='gd', help='Lower-level solver: gd|hb|fista')
    parser.add_argument('--ad_method', type=str, default='gd', help='Inexact AD method: gd|hb|cg')
    parser.add_argument('--ad_tol', type=float, default=0.01, help='Tolerance on residual for AD method')
    parser.add_argument('--upper_stepsize', type=float, default=0.01, help='Learning rate for upper-level GD iteration')

    args = parser.parse_args()
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.results_path, exist_ok=True)
    return args


def get_dataset(args):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset_train = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
    # Load data in with full batches rather than minibatching (for deterministic algorithm)
    train_sampler = torch.utils.data.sampler.SequentialSampler(dataset_train)
    test_sampler = torch.utils.data.sampler.SequentialSampler(dataset_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train), sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), sampler=test_sampler, **kwargs)
    # Loaders are set up to return full batch at each iteration, so just do this once and save to local variables
    train_data, train_labels = next(iter(train_loader))  # train_data size [60k, 1, 28, 28], train_labels size [60k]
    test_data, test_labels = next(iter(test_loader))  # test_data size [10k, 1, 28, 28], test_labels size [10k]
    return train_data[:args.training_size, ...], train_labels[:args.training_size], \
           test_data[:args.validation_size, ...], test_labels[:args.validation_size]


def make_corrupted_labels(labels, noise_rate, n_class):
    """
    Replace fraction noise_rate of labels with a new randomly generated (and fake) label
    """
    num_corrupted = int(noise_rate * (labels.size()[0]))
    randint = torch.randint(1, n_class, (num_corrupted,))
    index = torch.randperm(labels.size()[0])[:num_corrupted]
    labels[index] = (labels[index] + randint) % n_class
    return labels


def lower_obj(w, data_weights, train_data, train_corrupted_labels, args):
    output = torch.matmul(train_data, torch.t(w[:, :-1])) + w[:, -1]
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    loss = torch.nn.functional.cross_entropy(output, train_corrupted_labels, reduction='none')
    l2_reg = torch.pow(torch.linalg.vector_norm(w), 2)
    loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(data_weights))) + args.reg_value * l2_reg
    return loss_regu


def lower_obj_constants(data_weights, train_data, train_corrupted_labels, args):
    """
    Lipschitz and strong convexity constants for lower_obj wrt w (for a given value of data_weights)

    See cross_entropy_lipschiz.tex on Overleaf for derivation

    Note train_data doesn't have column of ones, so add this to row norms manually
    """
    train_data_row_sqnorms = torch.linalg.vector_norm(train_data, ord=2, dim=1)**2 + 1.0  # add column of ones
    L = float(torch.mean(torch.mul(train_data_row_sqnorms, torch.sigmoid(data_weights))) + 2*args.reg_value)
    mu = 2*args.reg_value  # from L2 reg
    return L, mu


def upper_obj(w, test_data, test_true_labels, args):
    output = torch.matmul(test_data, torch.t(w[:, :-1])) + w[:, -1]
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    loss = torch.nn.functional.cross_entropy(output, test_true_labels, reduction='mean')
    return loss


def lower_grad_inputs(w, data_weights, data, corrupted_labels, args):
    """
    Gradient of lower_obj w.r.t. w
    """
    # w_tmp = torch.tensor(w, requires_grad=True)
    w_tmp = w.clone().detach().requires_grad_(True)
    loss = lower_obj(w_tmp, data_weights, data, corrupted_labels, args)
    loss.backward()
    return w_tmp.grad.detach()


def lower_grad_parameters(w, data_weights, data, corrupted_labels, args):
    """
    Gradient of lower_obj w.r.t. data_weights
    """
    # data_weights_tmp = torch.tensor(data_weights, requires_grad=True)
    data_weights_tmp = data_weights.clone().detach().requires_grad_(True)
    loss = lower_obj(w, data_weights_tmp, data, corrupted_labels, args)
    loss.backward()
    return data_weights_tmp.grad.detach()


def lower_hess_vec(w, v, data_weights, data, corrupted_labels, args):
    """
    Hessian-vector product H*v for lower_obj (w.r.t. w)
    """
    assert v.shape == w.shape, "lower_hess_vec: w/v shape mismatch"
    eval_obj = lambda input: lower_obj(input, data_weights, data, corrupted_labels, args)
    return torch.autograd.functional.vhp(func=eval_obj, inputs=w, v=v)[1]


def lower_jac_vec(w, v, data_weights, data, corrupted_labels, args):
    """
    Given the Jacobian
        J(x,w) = d_w d_x lower_obj(x, w), of size |x|*|w|
    compute v.T * J(x,w) using the trick
        v.T * J(x,w) = grad_w [ grad_x(model)^T v ]
    where v is of size |x|.

    Notation: x above are the inputs (w) and w above are the upper-level parameters (data_weights)
    """
    assert v.shape == w.shape, "lower_jac_vec: v/w shape mismatch"
    w_tmp = w.clone().detach().requires_grad_(True)
    data_weights_tmp = data_weights.clone().detach().requires_grad_(True)
    output = lower_obj(w_tmp, data_weights_tmp, data, corrupted_labels, args)
    grad_lower_obj = torch.autograd.grad(output, w_tmp, create_graph=True)[0]
    dir_deriv = (grad_lower_obj * v).sum()  # computing dot product explicitly to avoid dimension mismatch issues
    dir_deriv.backward()
    return data_weights_tmp.grad.detach()


def upper_grad_inputs(w, test_data, test_true_labels, args):
    """
    Gradient of upper_obj w.r.t. w
    """
    w_tmp = w.clone().detach().requires_grad_(True)
    loss = upper_obj(w_tmp, test_data, test_true_labels, args)
    loss.backward()
    return w_tmp.grad.detach()


def fista(grad_fn, Y0, L, mu, niters=None, y0tol=None, ytol=None, verbose=False):
    """
    Solve this lower-level problem with FISTA
    Terminate after niters or ||Yk-Y*|| <= ytol
    y0tol is initial estimate ||Y0-Y*|| <= y0tol, if available (estimated otherwise)
    """
    copy = lambda v: v.detach().clone()
    norm = lambda v: float(torch.linalg.vector_norm(v, ord=2))
    dist_to_soln = lambda grad_at_Y: norm(grad_at_Y) / mu  # upper bound on ||Y-Y*|| given self.grad(Y)
    step_size = 1.0 / L
    q = step_size * mu
    w = 1.0 - sqrt(q)

    if niters is not None:
        # Run for fixed number of iterations, or ytol, if specified
        # assert ytol is None, "If niters is specified, ytol must be None"
        assert niters >= 0, "niters must be >= 0"
        max_iters = niters
        terminate_on_small_gradient = (ytol is not None)
    else:
        # Terminate when ||Yk-Y*|| <= ytol
        assert ytol is not None, "If niters is not specified, ytol must be given"
        assert ytol > 0.0, "ytol must be strictly positive"

        if y0tol is None:
            y0tol = dist_to_soln(grad_fn(Y0))

        max_iters = max(ceil(log(ytol**2 / ((L / mu) * (1 + sqrt(q)) * y0tol**2), w)), 0)
        # print("Will terminate lower-level FISTA after at most %g iters" % max_iters)
        terminate_on_small_gradient = True

    Y = copy(Y0)
    terminated_on_small_gradient = False

    # Main loop
    k = 0
    tk = 0.0  # initial value t0
    Ykm1 = copy(Y)  # Y_{-1}=Y0 initially
    while k < max_iters:
        tksq = tk ** 2
        tkp1 = (1 - q * tksq + sqrt((1 - q * tksq) ** 2 + 4 * tksq)) / 2
        beta_k = (tk - 1) * (1 - q * tkp1) / (tkp1 * (1 - q))
        Zk = Y + beta_k * (Y - Ykm1)
        Zk.add_(grad_fn(Zk), alpha=-step_size)  # Zk = Zk - step_size * self.gradient(Zk)
        # Update Y, Ykm1
        Ykm1 = copy(Y)
        Y = copy(Zk)
        # Update t
        tk = tkp1
        # Check termination
        k += 1
        g = grad_fn(Y)
        if verbose and k % 1000 == 0:
            print("  - FISTA iter %g, ||g|| = %g" % (k, norm(g)))
        # Check termination on xtol
        if terminate_on_small_gradient and dist_to_soln(g) <= ytol:
            terminated_on_small_gradient = True
            break
        # Quit on NaNs
        if torch.isnan(Y).any() or torch.isnan(g).any():
            break

    # Finalize
    grad = grad_fn(Y)
    return Y, norm(grad), k, terminated_on_small_gradient


def gradient_descent(grad_fn, Y0, L, mu, niters=None, y0tol=None, ytol=None, verbose=False):
    """
    Solve this lower-level problem with GD
    Terminate after niters or ||Yk-Y*|| <= ytol
    y0tol is initial estimate ||Y0-Y*|| <= y0tol, if available (estimated otherwise)
    """
    copy = lambda v: v.detach().clone()
    norm = lambda v: float(torch.linalg.vector_norm(v, ord=2))
    dist_to_soln = lambda grad_at_Y: norm(grad_at_Y) / mu  # upper bound on ||Y-Y*|| given grad(Y)
    step_size = 2.0 / (L + mu)
    qGD = (L - mu) / (L + mu)

    if niters is not None:
        # Run for fixed number of iterations, or ytol, if specified
        # assert ytol is None, "If niters is specified, ytol must be None"
        assert niters >= 0, "niters must be >= 0"
        max_iters = niters
        terminate_on_small_gradient = (ytol is not None)
    else:
        # Terminate when ||Yk-Y*|| <= ytol
        assert ytol is not None, "If niters is not specified, ytol must be given"
        assert ytol > 0.0, "ytol must be strictly positive"

        if y0tol is None:
            y0tol = dist_to_soln(grad_fn(Y0))

        # ytol <= y0tol * qGD**k so need k >= log(ytol/y0tol, qGD)
        max_iters = max(ceil(log(ytol / y0tol, qGD)), 0)
        # print("Will terminate lower-level GD after at most %g iters" % max_iters)
        terminate_on_small_gradient = True

    Y = copy(Y0)
    terminated_on_small_gradient = False

    # Main loop
    k = 0
    while k < max_iters:
        Y.add_(grad_fn(Y), alpha=-step_size)  # Y = Y - step_size * gradient(Y)
        # Check termination
        k += 1
        g = grad_fn(Y)
        if verbose and k % 1000 == 0:
            print("  - GD iter %g, ||g|| = %g" % (k, norm(g)))
        # Check termination on xtol
        if terminate_on_small_gradient and dist_to_soln(g) <= ytol:
            terminated_on_small_gradient = True
            break
        # Quit on NaNs
        if torch.isnan(Y).any() or torch.isnan(g).any():
            break

    # Finalize
    grad = grad_fn(Y)
    return Y, norm(grad), k, terminated_on_small_gradient


def heavy_ball(grad_fn, Y0, L, mu, niters=None, y0tol=None, ytol=None, verbose=False):
    """
    Solve this lower-level problem with HB
    Terminate after niters or ||Yk-Y*|| <= ytol
    y0tol is initial estimate ||Y0-Y*|| <= y0tol, if available (estimated otherwise)
    """
    copy = lambda v: v.detach().clone()
    norm = lambda v: float(torch.linalg.vector_norm(v, ord=2))
    dist_to_soln = lambda grad_at_Y: norm(grad_at_Y) / mu  # upper bound on ||Y-Y*|| given grad(Y)
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    heavy_ball_momentum = qHB ** 2  # beta
    step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha

    if niters is not None:
        # Run for fixed number of iterations, or ytol, if specified
        # assert ytol is None, "If niters is specified, ytol must be None"
        assert niters >= 0, "niters must be >= 0"
        max_iters = niters
        terminate_on_small_gradient = (ytol is not None)
    else:
        # Terminate when ||Yk-Y*|| <= ytol
        assert ytol is not None, "If niters is not specified, ytol must be given"
        assert ytol > 0.0, "ytol must be strictly positive"

        if y0tol is None:
            y0tol = dist_to_soln(grad_fn(Y0))

        # ytol <= y0tol * C* (qHB+gamma)**k so need k >= log(ytol/y0tol, qHB) if we take C=1, gamma=0 (assumed)
        max_iters = max(ceil(log(ytol / y0tol, qHB)), 0)
        # print("Will terminate lower-level HB after at most %g iters" % max_iters)
        terminate_on_small_gradient = True

    Y = copy(Y0)
    Ykm1 = copy(Y)  # Y_{-1}=Y0 initially
    terminated_on_small_gradient = False

    # Main loop
    k = 0
    while k < max_iters:
        Ynew = Y - step_size * grad_fn(Y) + heavy_ball_momentum * (Y - Ykm1)
        Ykm1 = copy(Y)
        Y = copy(Ynew)
        # Check termination
        k += 1
        g = grad_fn(Y)
        if verbose and k % 1000 == 0:
            print("  - HB iter %g, ||g|| = %g" % (k, norm(g)))
        # Check termination on xtol
        if terminate_on_small_gradient and dist_to_soln(g) <= ytol:
            terminated_on_small_gradient = True
            break
        # Quit on NaNs
        if torch.isnan(Y).any() or torch.isnan(g).any():
            break

    # Finalize
    grad = grad_fn(Y)
    return Y, norm(grad), k, terminated_on_small_gradient


def solve_lower_level(grad_fn, Y0, L, mu, method='gd', niters=None, y0tol=None, ytol=None, verbose=False):
    solver_lookup = {'gd': gradient_descent, 'hb': heavy_ball, 'fista': fista}
    if method.lower() in solver_lookup:
        Y, normg, niters_used, terminated_on_small_gradient = solver_lookup[method.lower()](grad_fn, Y0, L, mu,
                                                                                            niters=niters, y0tol=y0tol,
                                                                                            ytol=ytol, verbose=verbose)
    else:
        raise RuntimeError("Unknown lower-level solve '%s' (expect 'gd', 'hb' or 'fista')" % method)
    return Y, normg, niters_used, terminated_on_small_gradient


def inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, mu, L, niters, qtol):
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    step_size = 2.0 / (L + mu)

    # Use GD to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from zero
    q = torch.zeros(gradf.shape)
    r = hess_vec_prod(q) - gradf

    k = 0
    while k < niters and norm(r) > qtol:
        q.add_(r, alpha=-step_size)  # q = q - step_size * r
        r.add_(hess_vec_prod(r), alpha=-step_size)  # r = r - step_size * hess_vec_prod(r)
        k += 1

    # Calculate upper-level gradient from GD solution q
    upper_grad = -jac_vec_prod(q)
    return upper_grad, k


def inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, mu, L, niters, qtol):
    copy = lambda v: v.detach().clone()
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    heavy_ball_momentum = qHB ** 2  # beta
    step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha

    # Use HB to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from zero
    qprev = torch.zeros(gradf.shape)
    rprev = -gradf
    q = torch.zeros(gradf.shape)
    r = hess_vec_prod(q) - gradf

    k = 0
    while k < niters and norm(r) > qtol:
        qnew = q - step_size * r + heavy_ball_momentum * (q - qprev)
        rnew = r - step_size * hess_vec_prod(r) + heavy_ball_momentum * (r - rprev)

        qprev = copy(q)
        q = copy(qnew)
        rprev = copy(r)
        r = copy(rnew)

        k += 1

    # Calculate upper-level gradient from HB solution q
    upper_grad = -jac_vec_prod(q)
    return upper_grad, k


def inexact_ad_cg(gradf, hess_vec_prod, jac_vec_prod, mu, L, niters, qtol):
    copy = lambda v: v.detach().clone()
    dot = lambda v1, v2: (v1 * v2).sum()  # dot product
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    # Use CG to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from zero
    q = torch.zeros(gradf.shape)
    r = gradf - hess_vec_prod(q)
    d = copy(r)

    k = 0
    while k < niters and norm(r) > qtol:
        A_times_d = hess_vec_prod(d)
        alpha = dot(r, r) / dot(d, A_times_d)
        q.add_(d, alpha=alpha)  # q = q + alpha*d
        rnew = r - alpha * A_times_d
        beta = dot(rnew, rnew) / dot(r, r)
        d = rnew + beta * d
        r = copy(rnew)
        k += 1

    # Calculate upper-level gradient from CG solution q
    upper_grad = -jac_vec_prod(q)
    return upper_grad, k


def inexact_ad(gradf, hess_vec_prod, jac_vec_prod, mu, L, niters, qtol, method='ift'):
    ad_method_lookup = {'gd': inexact_ad_gd, 'hb': inexact_ad_hb, 'ift': inexact_ad_cg, 'cg': inexact_ad_cg}
    if method.lower() in ad_method_lookup:
        upper_grad, niters_used = ad_method_lookup[method.lower()](gradf, hess_vec_prod, jac_vec_prod, mu, L, niters, qtol)
    else:
        raise RuntimeError("Unknown inexact AD method '%s' (expect 'gd', 'hb' or 'ift')" % method)
    return upper_grad, niters_used


def run_solver():
    # Get dataset & solver parameters
    args = parse_args()

    # Load dataset
    torch.manual_seed(args.seed)
    train_data, train_labels, test_data, test_labels = get_dataset(args)
    train_data = torch.reshape(train_data, (train_data.size()[0], -1)).to(args.device)  # size [20k, 784]
    test_data = torch.reshape(test_data, (test_data.size()[0], -1)).to(args.device)  # size [5k, 784]
    train_labels = make_corrupted_labels(train_labels, args.noise_rate, args.num_classes).to(args.device)
    npixels = train_data.shape[1]

    # Construct wrappers to the important functions defined above
    my_lower_obj = lambda w_in, data_weights_in: lower_obj(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_obj_constants = lambda data_weights_in: lower_obj_constants(data_weights_in, train_data, train_labels, args)
    my_lower_grad_inputs = lambda w_in, data_weights_in: lower_grad_inputs(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_grad_parameters = lambda w_in, data_weights_in: lower_grad_inputs(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_hess_vec = lambda w_in, v_in, data_weights_in: lower_hess_vec(w_in, v_in, data_weights_in, train_data, train_labels, args)
    my_lower_jac_vec = lambda w_in, v_in, data_weights_in: lower_jac_vec(w_in, v_in, data_weights_in, train_data, train_labels, args)
    my_upper_obj = lambda w_in: upper_obj(w_in, test_data, test_labels, args)
    my_upper_grad = lambda w_in: upper_grad_inputs(w_in, test_data, test_labels, args)
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    # Initialize variables
    w = torch.zeros((args.num_classes, npixels + 1)).to(args.device)  # feature vector to learn in lower-level problem
    data_weights = torch.zeros((args.training_size,)).to(args.device)  # weights for each training data point (upper-level argument)
    upper_grad = torch.zeros(data_weights.shape).to(args.device)  # upper-level gradient

    # Data to save
    save_data = {}
    for col in ['k', 'cum_lower_iters', 'cum_ad_iters', 'cum_runtime', 'lower_iters', 'ad_iters', 'L', 'mu', 'upper_obj', 'lower_obj', 'norm_upper_grad']:
        save_data[col] = []

    filestem = f'budget_{args.total_budget}_lower_{args.lower_solver}_{args.lower_tol}_ad_{args.ad_method}_{args.ad_tol}_step_{args.upper_stepsize}'

    # Solve
    cumulative_lower_level_iters = 0
    cumulative_ad_iters = 0
    k = 0
    start_time = datetime.now()
    while cumulative_lower_level_iters + cumulative_ad_iters < args.total_budget:
        print(f"{filestem}: starting iteration {k} ({datetime.now():%Y-%m-%d %H:%M:%S})")

        # Lower-level solve (use previous minimizer w as a sensible starting point)
        grad_fn = lambda w_in: my_lower_grad_inputs(w_in, data_weights)
        L, mu = my_lower_obj_constants(data_weights)
        print(f"{filestem}: starting lower-level solve {k} ({datetime.now():%Y-%m-%d %H:%M:%S})")
        w, normg, niters_used, terminated_on_small_gradient = solve_lower_level(grad_fn, w, L, mu,
                                                                                method=args.lower_solver,
                                                                                ytol=args.lower_tol,
                                                                                niters=args.total_budget -(cumulative_lower_level_iters + cumulative_ad_iters),
                                                                                verbose=False)
        cumulative_lower_level_iters += niters_used

        if args.verbose:
            print(f"{filestem}: lower-level solve {k} has L={float(L)} and mu={float(mu)}, tol {args.lower_tol}")
            print(f"{filestem}: lower-level solve {k} finished after {niters_used} iters")

        # Run AD to get upper-level gradient (start from previous upper_grad as sensible starting point)
        gradf = my_upper_grad(w)
        hess_vec_prod = lambda v_in: my_lower_hess_vec(w, v_in, data_weights)
        jac_vec_prod = lambda v_in: my_lower_jac_vec(w, v_in, data_weights)
        print(f"{filestem}: starting inexact AD {k} ({datetime.now():%Y-%m-%d %H:%M:%S})")
        upper_grad, ad_niters_used = inexact_ad(gradf, hess_vec_prod, jac_vec_prod, mu, L,
                                                args.total_budget -(cumulative_lower_level_iters + cumulative_ad_iters),
                                                args.ad_tol,
                                                method=args.ad_method)
        cumulative_ad_iters += ad_niters_used

        if args.verbose:
            print(f"{filestem}: inexact AD {k} finished after {ad_niters_used} iters")
            print(f"{filestem}: gradf has norm {float(norm(gradf))} and upper_grad has norm {float(norm(upper_grad))}")

        # Save data
        save_data['k'].append(k)
        save_data['cum_lower_iters'].append(cumulative_lower_level_iters)
        save_data['cum_ad_iters'].append(cumulative_ad_iters)
        save_data['cum_runtime'].append((datetime.now() - start_time).total_seconds())
        save_data['lower_iters'].append(niters_used)
        save_data['ad_iters'].append(ad_niters_used)
        save_data['L'].append(L)
        save_data['mu'].append(mu)
        save_data['upper_obj'].append(float(my_upper_obj(w)))
        save_data['lower_obj'].append(float(my_lower_obj(w, data_weights)))
        save_data['norm_upper_grad'].append(norm(upper_grad))

        # Take upper-level GD step
        data_weights.add_(upper_grad, alpha=-args.upper_stepsize)  # data_weights = data_weights - stepsize * upper_grad
        k += 1

    if args.save_results:
        # Save optimization history
        df = pd.DataFrame.from_dict(save_data)
        df.to_csv(os.path.join(args.results_path, filestem + '.csv'), index=False)

        # Save final results
        torch.save(data_weights, os.path.join(args.results_path, filestem + '_data_weights.pt'))
        torch.save(w, os.path.join(args.results_path, filestem + '_w.pt'))
    return


def test_cg():
    # It seems that CG is giving very different results to HB as an AD method so try to check it here
    # Get dataset & solver parameters
    args = parse_args()

    # Override some defaults for more useful checking
    lower_solver = 'hb'
    lower_tol = 0.1
    upper_stepsize = 0.1
    ad_niters = 100

    # Load dataset
    torch.manual_seed(args.seed)
    train_data, train_labels, test_data, test_labels = get_dataset(args)
    train_data = torch.reshape(train_data, (train_data.size()[0], -1)).to(args.device)  # size [20k, 784]
    test_data = torch.reshape(test_data, (test_data.size()[0], -1)).to(args.device)  # size [5k, 784]
    train_labels = make_corrupted_labels(train_labels, args.noise_rate, args.num_classes).to(args.device)
    npixels = train_data.shape[1]

    # Construct wrappers to the important functions defined above
    my_lower_obj = lambda w_in, data_weights_in: lower_obj(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_obj_constants = lambda data_weights_in: lower_obj_constants(data_weights_in, train_data, train_labels, args)
    my_lower_grad_inputs = lambda w_in, data_weights_in: lower_grad_inputs(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_grad_parameters = lambda w_in, data_weights_in: lower_grad_inputs(w_in, data_weights_in, train_data, train_labels, args)
    my_lower_hess_vec = lambda w_in, v_in, data_weights_in: lower_hess_vec(w_in, v_in, data_weights_in, train_data, train_labels, args)
    my_lower_jac_vec = lambda w_in, v_in, data_weights_in: lower_jac_vec(w_in, v_in, data_weights_in, train_data, train_labels, args)
    my_upper_obj = lambda w_in: upper_obj(w_in, test_data, test_labels, args)
    my_upper_grad = lambda w_in: upper_grad_inputs(w_in, test_data, test_labels, args)
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    # Initialize variables
    w = torch.zeros((args.num_classes, npixels + 1)).to(args.device)  # feature vector to learn in lower-level problem
    data_weights = torch.zeros((args.training_size,)).to(args.device)  # weights for each training data point (upper-level argument)
    upper_grad = torch.zeros(data_weights.shape).to(args.device)  # upper-level gradient

    # Solve
    cumulative_lower_level_iters = 0
    cumulative_ad_iters = 0
    k = 0
    while k < 4:
        # Lower-level solve (use previous minimizer w as a sensible starting point)
        print("**** Iteration %g ****" % k)
        grad_fn = lambda w_in: my_lower_grad_inputs(w_in, data_weights)
        L, mu = my_lower_obj_constants(data_weights)
        print("Iter %g: starting lower-level solve using %s and tol %g" % (k, lower_solver, lower_tol))
        w, normg, niters_used, terminated_on_small_gradient = solve_lower_level(grad_fn, w, L, mu,
                                                                                method=lower_solver,
                                                                                ytol=lower_tol,
                                                                                verbose=False)
        cumulative_lower_level_iters += niters_used

        # Run AD to get upper-level gradient (start from previous upper_grad as sensible starting point)
        gradf = my_upper_grad(w)
        hess_vec_prod = lambda v_in: my_lower_hess_vec(w, v_in, data_weights)
        jac_vec_prod = lambda v_in: my_lower_jac_vec(w, v_in, data_weights)
        upper_grad_ift, ad_niters_used = inexact_ad(gradf, hess_vec_prod, jac_vec_prod, mu, L, ad_niters, qtol=0.01, method='ift')
        upper_grad_hb, ad_niters_used_hb = inexact_ad(gradf, hess_vec_prod, jac_vec_prod, mu, L, ad_niters, qtol=0.01, method='hb')
        upper_grad = upper_grad_ift
        cumulative_ad_iters += ad_niters_used

        print("CG used %g iters, HB used %g iters" % (ad_niters_used, ad_niters_used_hb))
        print("||upper_grad_cg|| = %g" % norm(upper_grad))
        print("||upper_grad_hb|| = %g" % norm(upper_grad_hb))
        print("||ift - hb|| = %g" % norm(upper_grad - upper_grad_hb))
        # print("||(-ift) - hb|| = %g" % norm((-upper_grad) - upper_grad_hb))
        # print("||(2*ift) - hb|| = %g" % norm((2*upper_grad) - upper_grad_hb))
        print("Taking upper step using CG gradient")

        # Take upper-level GD step
        data_weights.add_(upper_grad, alpha=-upper_stepsize)  # data_weights = data_weights - stepsize * upper_grad
        k += 1
    print("Done")
    return


def main():
    # Run a single test (based on sys.argv)
    run_solver()
    # test_cg()  # try to find issue with AD routines
    return


if __name__ == '__main__':
    main()
