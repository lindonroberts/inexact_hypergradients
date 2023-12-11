#!/usr/bin/env python3

"""
Test of error bounds on a simple artificial problem with analytic solutions for everything

Problem taken from Section 6.1 of [1]:

min_{w} f(w) := ||A*x(w)-b||_2^2
s.t. x(w) = argmin_{x} ||C*w+D*x-e||_2^2

Next version with updated AD bounds (a priori and a posteriori)

[1] J. Li, B. Gu, H. Huang. A Fully Single Loop Algorithm for Bilevel Optimization without Hessian Inverse,
Proceedings of the AAAI Conference on Artificial Intelligence, 2022.
"""
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# Lower-level solvers still fine, just need to update AD methods
from quadratic_problem_example import lstsq, sumsq, QuadraticProblem, fista, gradient_descent, heavy_ball

# For FISTA/CG, GD, HB
# COLORS = ['black', 'gray', 'darkgray']  # IMA paper
COLORS = ['C0', 'C1', 'C2']  # arxiv
# COLORS = ['#d7301f', '#fc8d59', '#fdcc8a']
# COLORS = ['#5e3c99', '#e66101', '#b2abd2']  # dark purple, red, light purple


def inexact_ad_cg(gradf, hess_vec_prod, jac_vec_prod, g0, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    # Use CG to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from g0
    q = g0.copy()
    r = gradf - hess_vec_prod(q)
    d = r.copy()

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        upper_grad = -jac_vec_prod(q)
        resid_norm = np.linalg.norm(r)
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    for k in range(niters):
        if np.linalg.norm(r) > 0.0:  # don't run CG while residual is too small
            A_times_d = hess_vec_prod(d)
            alpha = np.dot(r, r) / np.dot(d, A_times_d)
            q = q + alpha * d
            rnew = r - alpha * A_times_d
            beta = np.dot(rnew, rnew) / np.dot(r, r)
            d = rnew + beta * d
            r = rnew.copy()
        # otherwise (||r|| small), do nothing (to avoid NaN errors)

        if Gtrue is not None:
            upper_grad = -jac_vec_prod(q)
            resid_norm = np.linalg.norm(r)
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    # Calculate upper-level gradient from CG solution q
    upper_grad = -jac_vec_prod(q)
    if Gtrue is not None:
        for key in Gerr:
            Gerr[key] = np.array(Gerr[key])
        return upper_grad, Gerr
    else:
        return upper_grad


def inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    # Use GD to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from g0
    q = g0.copy()
    r = hess_vec_prod(q) - gradf
    step_size = 2.0 / (L + mu)

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        upper_grad = -jac_vec_prod(q)
        resid_norm = np.linalg.norm(r)
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    for k in range(niters):
        q = q - step_size * r
        r = r - step_size * hess_vec_prod(r)

        if Gtrue is not None:
            upper_grad = -jac_vec_prod(q)
            resid_norm = np.linalg.norm(r)
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    # Calculate upper-level gradient from CG solution q
    upper_grad = -jac_vec_prod(q)
    if Gtrue is not None:
        for key in Gerr:
            Gerr[key] = np.array(Gerr[key])
        return upper_grad, Gerr
    else:
        return upper_grad


def inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    # Use HB to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from g0
    qprev = np.zeros(g0.shape)
    rprev = -gradf.copy()
    q = g0.copy()
    r = hess_vec_prod(q) - gradf
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    heavy_ball_momentum = qHB ** 2  # beta
    step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        upper_grad = -jac_vec_prod(q)
        resid_norm = np.linalg.norm(r)
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    for k in range(niters):
        qnew = q - step_size * r + heavy_ball_momentum * (q - qprev)
        rnew = r - step_size * hess_vec_prod(r) + heavy_ball_momentum * (r - rprev)
        q, qprev = qnew, q
        r, rprev = rnew, r

        if Gtrue is not None:
            upper_grad = -jac_vec_prod(q)
            resid_norm = np.linalg.norm(r)
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(resid_norm))

    # Calculate upper-level gradient from CG solution q
    upper_grad = -jac_vec_prod(q)
    if Gtrue is not None:
        for key in Gerr:
            Gerr[key] = np.array(Gerr[key])
        return upper_grad, Gerr
    else:
        return upper_grad


def compare_lower_level_solvers(font_size='large', fmt='png'):
    """
    Compare convergence rate of FISTA/GD/HB for the lower-level problem xk(w) -> x(w), and associated error bounds
    """
    prob = QuadraticProblem(m1=1000, m2=1000, nx=10, nw=10)
    w = np.ones((prob.nw,))
    x_w = prob.lower_soln(w)
    x0 = np.zeros((prob.nx,))
    niters = 100
    grad_fn = lambda x: prob.lower_grad_inputs(x, w)
    L, mu = prob.lip_const(w), prob.convex_const(w)

    _, fista_errs = fista(grad_fn, x0, L, mu, niters, Ytrue=x_w)
    _, gd_errs = gradient_descent(grad_fn, x0, L, mu, niters, Ytrue=x_w)
    _, hb_errs = heavy_ball(grad_fn, x0, L, mu, niters, Ytrue=x_w)

    plt.figure(1)
    plt.clf()
    plt.semilogy(fista_errs['true'], color=COLORS[0], marker='o', linestyle='-', label='FISTA', linewidth=2, markevery=10)
    plt.semilogy(fista_errs['apriori'], color=COLORS[0], marker='o', linestyle='--', label='FISTA bound', linewidth=2, markevery=10)
    plt.semilogy(gd_errs['true'], color=COLORS[1], marker='^', linestyle='-', label='GD', linewidth=2, markevery=10)
    plt.semilogy(gd_errs['apriori'], color=COLORS[1], marker='^', linestyle='--', label='GD bound', linewidth=2, markevery=10)
    plt.semilogy(hb_errs['true'], color=COLORS[2], marker='s', linestyle='-', label='HB', linewidth=2, markevery=10)
    plt.semilogy(hb_errs['apriori'], color=COLORS[2], marker='s', linestyle='--', label='HB bound*', linewidth=2, markevery=10)  # HB a priori bound is not really
    ax = plt.gca()
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Iterate Error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='lower left', fontsize=font_size, fancybox=True)
    plt.ylim(1e-16, 1e2)
    plt.grid()

    plt.savefig('quadratic_problem_plots_v2/compare_lower_level_apriori.%s' % fmt, bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.semilogy(fista_errs['true'], color=COLORS[0], marker='o', linestyle='-', label='FISTA', linewidth=2, markevery=10)
    plt.semilogy(fista_errs['aposteriori'], color=COLORS[0], marker='o', linestyle='--', label='FISTA bound', linewidth=2, markevery=10)
    plt.semilogy(gd_errs['true'], color=COLORS[1], marker='^', linestyle='-', label='GD', linewidth=2, markevery=10)
    plt.semilogy(gd_errs['aposteriori'], color=COLORS[1], marker='^', linestyle='--', label='GD bound', linewidth=2, markevery=10)
    plt.semilogy(hb_errs['true'], color=COLORS[2], marker='s', linestyle='-', label='HB', linewidth=2, markevery=10)
    plt.semilogy(hb_errs['aposteriori'], color=COLORS[2], marker='s', linestyle='--', label='HB bound', linewidth=2, markevery=10)

    ax = plt.gca()
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Iterate Error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='lower left', fontsize=font_size, fancybox=True)
    plt.grid()
    plt.ylim(1e-16, 1e2)

    plt.savefig('quadratic_problem_plots_v2/compare_lower_level_aposteriori.%s' % fmt, bbox_inches='tight')
    plt.show()
    return


def compare_inexact_ad(lower_solve_type, font_size='large', fmt='png', ylim=None):
    prob = QuadraticProblem(m1=1000, m2=1000, nx=10, nw=10)
    w = np.ones((prob.nw,))
    L, mu = prob.lip_const(w), prob.convex_const(w)

    # Do lower-level solve to get approximate x_w
    if lower_solve_type == 'exact':
        # Use true lower-level solution x(w) for now
        print("Using exact lower-level solution")
        x_w = prob.lower_soln(w)  # + 1e-15 * np.ones((prob.nx,))
    elif lower_solve_type.startswith('hb'):  # hbN for N iterations of HB (the fastest lower-level method for this problem)
        x0 = np.zeros((prob.nx,))
        niters = int(lower_solve_type.replace('hb', ''))
        print("Using %g iterations of HB to get lower-level solution" % niters)
        x_w = heavy_ball(lambda x: prob.lower_grad_inputs(x, w), x0, L, mu, niters)
    else:
        raise RuntimeError("compare_inexact_ad: unknown lower_solve_type '%s'" % lower_solve_type)

    gradf = prob.upper_grad_inputs(x_w, w)
    hess_vec_prod = lambda v: prob.lower_hess_vec_inputs(x_w, w, v)
    jac_vec_prod = lambda v: prob.lower_jac_vec(x_w, w, v)
    g0 = np.zeros((prob.nw,))
    niters = 200
    # Gtrue = -prob.lower_jac(x_w, w).T @ np.linalg.solve(prob.lower_hess_inputs(x_w, w), gradf)  # full gradient using above x(w) value
    Gtrue = prob.upper_grad_soln(w)  # full gradient at w using exact x(w)

    # Constants used in bounds
    Bnorm = lambda x: np.linalg.norm(prob.lower_jac(x, w), ord=2)  # Bnorm(x) = ||B(x)||
    norm_gradf = lambda x: np.linalg.norm(prob.upper_grad_inputs(x, w))
    Lf = prob.lip_upper()
    LB = prob.lip_jac()
    x_w_true = prob.lower_soln(w)  # x*(w) in paper notation
    eps = np.linalg.norm(x_w - x_w_true)  # true error
    print("Lower-level solve error ||x_w - xtrue|| = %g" % eps)
    norm_gradf_x_w = norm_gradf(x_w)  # np.linalg.norm(prob.upper_grad_inputs(x_w, w))
    norm_gradf_xtrue = norm_gradf(x_w_true)  # np.linalg.norm(prob.upper_grad_inputs(x_w_true, w))
    qGD = (L - mu) / (L + mu)
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    C_HB, gamma_HB = 1.0, 0.0  # we don't know these values, so just pick something
    norm_gradg_x_w = np.linalg.norm(prob.lower_grad_inputs(x_w, w))  # ||grad g(x_w)||

    def apriori_delta(k, method):
        """
        A priori estimate of delta after k iterations

        Based on estimates of ||qk-q*|| in terms of ||q0-q*|| from different sources.

        Then convert to bound on ||A*qk - gradf|| <= const * ||A*q0 - gradf|| using Rayleigh quotient
        (introduces an extra L/mu factor in const)
        and then use q0=0 to simplify RHS
        """
        init_residual = norm_gradf_xtrue + Lf * eps  # start from q0=0, so r=grad f(x_w), adjusted to be a priori
        if method == 'gd':
            # From Mehmood & Ochs (2020), Lemma 6
            return (L/mu) * (qGD ** k) * init_residual
        elif method == 'hb':
            # From Mehmood & Ochs (2020), Lemma 13
            return (L/mu) * C_HB * ((qHB + gamma_HB) ** k) * init_residual
        elif method == 'cg':
            # From Nocedal & Wright (2006), eq (5.36)
            # That bound measures ||qk-q*||_A, which we then adjust using
            # sqrt(mu)*||qk-q*|| <= ||qk-q*||_A <= sqrt(L) * ||qk-q*||
            return 2 * (L / mu) ** 1.5 * (qHB ** k) * (norm_gradf_xtrue + Lf * eps)
        else:
            raise RuntimeError("apriori_delta: unknown method '%s'" % method)

    def apriori_est(k, method):
        # using exact value for epsilon for demonstration purposes
        delta = apriori_delta(k, method=method)  # as measured using linear rates
        return (Bnorm(x_w_true) * delta / mu) \
               + (LB * delta * eps / mu) \
               + (norm_gradf_xtrue * LB * eps / mu) \
               + (Lf * LB * eps**2 / mu) \
               + (Bnorm(x_w_true) * Lf * eps / mu)

    def aposteriori_est(rnorm):  # depends on current residual rnorm = ||A*qk - gradf||_2
        return (Bnorm(x_w) * rnorm / mu) \
               + (norm_gradf_x_w * LB * norm_gradg_x_w / mu**2) \
               + (Bnorm(x_w) * Lf * norm_gradg_x_w / mu**2) \
               + (LB * Lf * norm_gradg_x_w**2 / mu**3)

    _, cg_errs = inexact_ad_cg(gradf, hess_vec_prod, jac_vec_prod, g0, niters, Gtrue=Gtrue,
                                apriori_est=lambda k: apriori_est(k, method='cg'), aposteriori_est=aposteriori_est)
    _, gd_errs = inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=Gtrue,
                                apriori_est=lambda k: apriori_est(k, method='gd'), aposteriori_est=aposteriori_est)
    _, hb_errs = inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=Gtrue,
                               apriori_est=lambda k: apriori_est(k, method='hb'), aposteriori_est=aposteriori_est)

    plt.figure(1)
    plt.clf()
    plt.semilogy(cg_errs['true'], color=COLORS[0], marker='o', linestyle='-', label='CG', linewidth=2, markevery=20)
    plt.semilogy(cg_errs['apriori'], color=COLORS[0], marker='o', linestyle='--', label='CG bound', linewidth=2, markevery=20)
    plt.semilogy(gd_errs['true'], color=COLORS[1], marker='^', linestyle='-', label='GD', linewidth=2, markevery=20)
    plt.semilogy(gd_errs['apriori'], color=COLORS[1], marker='^', linestyle='--', label='GD bound', linewidth=2, markevery=20)
    plt.semilogy(hb_errs['true'], color=COLORS[2], marker='s', linestyle='-', label='HB', linewidth=2, markevery=20)
    plt.semilogy(hb_errs['apriori'], color=COLORS[2], marker='s', linestyle='--', label='HB bound*', linewidth=2, markevery=20)  # HB a priori bound not actually real
    ax = plt.gca()
    ax.set_xlabel("AD Iterations", fontsize=font_size)
    ax.set_ylabel('Gradient Error', fontsize=font_size)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size, fancybox=True)
    plt.grid()

    plt.savefig('quadratic_problem_plots_v2/compare_inexact_ad_%s_apriori.%s' % (lower_solve_type, fmt), bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.semilogy(cg_errs['true'], color=COLORS[0], marker='o', linestyle='-', label='CG', linewidth=2, markevery=20)
    plt.semilogy(cg_errs['aposteriori'], color=COLORS[0], marker='o', linestyle='--', label='CG bound', linewidth=2, markevery=20)
    plt.semilogy(gd_errs['true'], color=COLORS[1], marker='^', linestyle='-', label='GD', linewidth=2, markevery=20)
    plt.semilogy(gd_errs['aposteriori'], color=COLORS[1], marker='^', linestyle='--', label='GD bound', linewidth=2, markevery=20)
    plt.semilogy(hb_errs['true'], color=COLORS[2], marker='s', linestyle='-', label='HB', linewidth=2, markevery=20)
    plt.semilogy(hb_errs['aposteriori'], color=COLORS[2], marker='s', linestyle='--', label='HB bound', linewidth=2, markevery=20)
    ax = plt.gca()
    ax.set_xlabel("AD Iterations", fontsize=font_size)
    ax.set_ylabel('Gradient Error', fontsize=font_size)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size, fancybox=True)
    plt.grid()

    plt.savefig('quadratic_problem_plots_v2/compare_inexact_ad_%s_aposteriori.%s' % (lower_solve_type, fmt), bbox_inches='tight')
    plt.show()
    return


def fixed_iter_comparison(ad_method, hb_iters_list, font_size='large', fmt='png', ylim=None):
    """
    The results from compare_inexact_ad with type 'hbN' shows that the different choices of N give nice error comparisons.

    Here, show the true error using a given AD method, adjusted for total work.
    """
    prob = QuadraticProblem(m1=1000, m2=1000, nx=10, nw=10)
    w = np.ones((prob.nw,))
    L, mu = prob.lip_const(w), prob.convex_const(w)

    # Do lower-level solve to get approximate x_w
    x0 = np.zeros((prob.nx,))

    plt.figure(1)
    plt.clf()

    for hb_niters, col, mkr in hb_iters_list:
        print("Using %g iterations of HB to get lower-level solution" % hb_niters)
        x_w = heavy_ball(lambda x: prob.lower_grad_inputs(x, w), x0, L, mu, hb_niters)

        gradf = prob.upper_grad_inputs(x_w, w)
        hess_vec_prod = lambda v: prob.lower_hess_vec_inputs(x_w, w, v)
        jac_vec_prod = lambda v: prob.lower_jac_vec(x_w, w, v)
        g0 = np.zeros((prob.nw,))
        ad_niters = 200 - hb_niters
        Gtrue = prob.upper_grad_soln(w)  # full gradient at w using exact x(w)

        def apriori_est(k, method):
            return None  # not used, so can choose anything

        def aposteriori_est(rnorm):  # depends on current residual rnorm = ||A*qk - gradf||_2
            return None  # not used, so can choose anything

        if ad_method == 'cg' or ad_method == 'ift':
            _, ad_errs = inexact_ad_cg(gradf, hess_vec_prod, jac_vec_prod, g0, ad_niters, Gtrue=Gtrue,
                                       apriori_est=lambda k: apriori_est(k, method='cg'), aposteriori_est=aposteriori_est)
        elif ad_method == 'gd':
            _, ad_errs = inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, ad_niters, Gtrue=Gtrue,
                                       apriori_est=lambda k: apriori_est(k, method='gd'), aposteriori_est=aposteriori_est)
        elif ad_method == 'hb':
            _, ad_errs = inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, ad_niters, Gtrue=Gtrue,
                                       apriori_est=lambda k: apriori_est(k, method='hb'), aposteriori_est=aposteriori_est)
        else:
            raise RuntimeError("fixed_iter_comparison: Unknown ad_method '%s'" % ad_method)

        # Pad start with initial error value so overall length is 200
        plt.semilogy(np.pad(ad_errs['true'], (hb_niters, 0), 'edge'), '-', label='N = %g' % hb_niters, linewidth=2, color=col, marker=mkr, markevery=20)

    ax = plt.gca()
    ax.set_xlabel("Total lower-level/AD iterations", fontsize=font_size)
    ax.set_ylabel('Gradient Error', fontsize=font_size)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='best', fontsize=font_size, fancybox=True)
    plt.grid()

    plt.savefig('quadratic_problem_plots_v2/compare_inexact_ad_%s_vary_lower_solve.%s' % (ad_method, fmt), bbox_inches='tight')
    return


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font_size = 'x-large'
    fmt = 'pdf'

    compare_lower_level_solvers(font_size=font_size, fmt=fmt)
    for lower_solve_type in ['exact'] + ['hb%g' % N for N in [20, 40, 60, 80, 100]]:
        if lower_solve_type == 'exact':
            ylim = (1e-13, 1e8)
        elif '20' in lower_solve_type:
            ylim = (1, 1e8)
        elif '60' in lower_solve_type:
            ylim = (1e-6, 1e8)
        elif '100' in lower_solve_type:
            ylim = (1e-12, 1e8)
        else:
            ylim = None
        compare_inexact_ad(lower_solve_type, font_size=font_size, fmt=fmt, ylim=ylim)
    for ad_method in ['hb', 'cg']:  #['gd', 'hb', 'cg']:
        # iters_list = [(20, 'black', 'o'), (40, 'dimgray', '^'), (60, 'gray', 's'), (80, 'darkgray', 'P'), (100, 'silver', 'D')]  # IMA paper
        iters_list = [(20, 'C0', 'o'), (40, 'C1', '^'), (60, 'C2', 's'), (80, 'C3', 'P'), (100, 'C4', 'D')]  # arxiv
    #     iters_list = [(20, '#b30000', 'o'), (40, '#e34a33', '^'), (60, '#fc8d59', 's'), (80, '#fdcc8a', 'P'), (100, '#fef0d9', 'D')]
        fixed_iter_comparison(ad_method, iters_list, font_size=font_size, fmt=fmt, ylim=(1e-9, 2e4))

    print("Done")
    return


if __name__ == '__main__':
    main()
