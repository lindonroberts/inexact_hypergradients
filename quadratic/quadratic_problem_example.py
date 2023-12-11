#!/usr/bin/env python3

"""
Test of error bounds on a simple artificial problem with analytic solutions for everything

Problem taken from Section 6.1 of [1]:

min_{w} f(w) := ||A*x(w)-b||_2^2
s.t. x(w) = argmin_{x} ||C*w+D*x-e||_2^2

[1] J. Li, B. Gu, H. Huang. A Fully Single Loop Algorithm for Bilevel Optimization without Hessian Inverse,
Proceedings of the AAAI Conference on Artificial Intelligence, 2022.
"""
from math import ceil, log, sqrt
import numpy as np
import matplotlib.pyplot as plt


def lstsq(A, b):
    """
    Solve the linear least-squares problem: min_{x} ||Ax-b||_2^2.

    Just a simple wrapper to NumPy, but done this way because we need it in several places below.
    """
    return np.linalg.lstsq(A, b, rcond=None)[0]


def sumsq(x):
    """
    Fast computation of ||x||_2^2
    """
    return np.dot(x, x)


class QuadraticProblem(object):
    """
    Wrapper class which implements the various problem features (gradients, Hessian-vector products, etc.)
    """
    def __init__(self, m1=10000, m2=10000, nx=5, nw=5, noise_stdev=0.01, seed=0):
        """
        Create a test problem of a given size/noise level using the method from [1].
        """
        self.m1 = m1
        self.m2 = m2
        self.nx = nx
        self.nw = nw
        self.noise_stdev = noise_stdev
        self.seed = seed
        # Initialize problem
        np.random.seed(self.seed)
        self.A = np.random.uniform(size=(self.m1, self.nx))
        self.b = self.A @ np.random.uniform(size=(self.nx,)) + self.noise_stdev * np.random.normal(size=(self.m1,))
        self.C = np.random.uniform(size=(self.m2, self.nw))
        self.D = np.random.uniform(size=(self.m2, self.nx))
        self.e = self.C @ np.random.uniform(size=(self.nw,)) + self.D @ np.random.uniform(size=(self.nx,)) \
                 + self.noise_stdev * np.random.normal(size=(self.m2,))

        # Calculate some useful quantities
        self.Anorm = np.linalg.norm(self.A, ord=2)
        D_sing_vals = np.linalg.svd(self.D, compute_uv=False)
        self.L = 2 * np.max(D_sing_vals)**2  # lower-level gradient Lipschitz constant
        self.mu = 2 * np.min(D_sing_vals)**2  # lower-level strong convexity constant
        self.Dpinv = np.linalg.pinv(self.D)  # size nx*m2
        self.jac_norm = 2 * np.linalg.norm(self.D.T @ self.C, ord=2)  # bound on lower-level Jacobian = 2 @ D.T @ C
        self.upper_soln_matrix = self.A @ self.Dpinv @ self.C  # (m1*nx) (nx*m2) * (m2*nw) = (m1*nw)
        return

    def lip_const(self, w: np.ndarray):
        """
        Lipschitz constant of lower-level gradient (for a given w)

        Note: constant, doesn't depend on w
        """
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return self.L  # pre-computed in self.__init__()

    def convex_const(self, w: np.ndarray):
        """
        Strong convexity constant of lower-level problem (for a given w)

        Note: constant, doesn't depend on w
        """
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return self.mu  # pre-computed in self.__init__()

    def lip_upper(self):
        """
        Lipschitz constant of the upper-level gradient (wrt x)
        """
        return 2 * self.Anorm**2

    def lip_hess(self):
        """
        Bound on the Lipschitz constant of the lower-level Hessian (w.r.t x), which must hold for all x and w
        """
        return 0.0  # lower-level Hessian is constant for this problem

    def lip_jac(self):
        """
        Bound on the Lipschitz constant of the lower-level Jacobian, which must hold for all x and w
        """
        return 0.0  # lower-level Jacobian is constant for this problem

    def jac_bound(self):
        """
        Bound on the norm of the lower-level Jacobian, which must hold for all x and w
        """
        return self.jac_norm  # pre-computed in self.__init__()

    def lower_obj(self, x: np.ndarray, w: np.ndarray):
        """
        Evaluate lower-level objective for a given x and w
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return sumsq(self.C @ w + self.D @ x - self.e)

    def lower_grad_params(self, x: np.ndarray, w: np.ndarray):
        """
        Evaluate lower-level gradient w.r.t. parameters w (for a given x and w)
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return 2 * self.C.T @ (self.C @ w + self.D @ x - self.e)

    def lower_grad_inputs(self, x: np.ndarray, w: np.ndarray):
        """
        Evaluate lower-level gradient w.r.t. argument x (for a given x and w)
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return 2 * self.D.T @ (self.C @ w + self.D @ x - self.e)

    def lower_hess_vec_inputs(self, x: np.ndarray, w: np.ndarray, v: np.ndarray):
        """
        Calculate Hessian vector product H(x,w) @ v, where H(x,w) is the lower-level Hessian (w.r.t. argument x)
            H(x,w) = 2 * D.T @ D

        Note that v must be of size |x|
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        assert v.shape == (self.nx,), "v has wrong shape (got %s, expect (%g,))" % (v.shape, self.nx)
        return 2 * self.D.T @ (self.D @ v)

    def lower_hess_inputs(self, x: np.ndarray, w: np.ndarray):
        """
        Lower-level Hessian (wrt input x)
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return 2 * self.D.T @ self.D

    def lower_jac_vec(self, x: np.ndarray, w: np.ndarray, v: np.ndarray):
        """
        Calculate Jacobian-vector product v.T @ J(x,w), where J(x,w) is the lower-level Jacobian:
            J(x, w) = d_w d_x lower_obj(x,w)
        of size |x|*|w|.

        Note that v must be of size |x|

        Here, J(x,w) = 2 * D.T @ C (to get correct dimensions |x|*|w|)
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        assert v.shape == (self.nx,), "v has wrong shape (got %s, expect (%g,))" % (v.shape, self.nx)
        # return 2 * v.T @ self.D.T @ self.C
        return 2 * self.C.T @ (self.D @ v)

    def lower_jac(self, x: np.ndarray, w: np.ndarray):
        """
        Lower-level Jacobian J(x,w), of size |x|*|w|
        """
        assert x.shape == (self.nx,), "x has wrong shape (got %s, expect (%g,))" % (x.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return 2 * self.D.T @ self.C

    def upper_obj(self, x_w: np.ndarray, w: np.ndarray):
        """
        Evaluate upper-level objective for a given x(w) and w

        Note: here, x_w is the approximate solution x(w) to the lower-level problem with w
        """
        assert x_w.shape == (self.nx,), "x_w has wrong shape (got %s, expect (%g,))" % (x_w.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return sumsq(self.A @ x_w - self.b)

    def upper_grad_inputs(self, x_w: np.ndarray, w: np.ndarray):
        """
        Gradient of upper-level objective for a given x(w) and w, w.r.t. x_w

        Note: here, x_w is the approximate solution x(w) to the lower-level problem with w
        """
        assert x_w.shape == (self.nx,), "x_w has wrong shape (got %s, expect (%g,))" % (x_w.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return 2 * self.A.T @ (self.A @ x_w - self.b)

    def upper_grad_params(self, x_w: np.ndarray, w: np.ndarray):
        """
        Gradient of upper-level objective for a given x(w) and w, w.r.t. w directly (i.e. not through x_w)

        Note: here, x_w is the approximate solution x(w) to the lower-level problem with w
        """
        assert x_w.shape == (self.nx,), "x_w has wrong shape (got %s, expect (%g,))" % (x_w.shape, self.nx)
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return np.zeros((self.nw,))  # no direct dependency on w in upper-level objective

    def lower_soln(self, w: np.ndarray):
        """
        The true lower-level minimizer x(w), used for validation only
        """
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return lstsq(self.D, self.e - self.C @ w)

    def lower_soln_grad_params(self, w: np.ndarray):
        """
        Gradient of x(w) wrt w, used for validation only, of size |x|*|w|
        """
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        return -self.Dpinv @ self.C

    def upper_soln(self):
        """
        The true upper minimizer w*, used for validation only

        Since x(w) = Dpinv @ (e - C @ w), we have the LLS upper-level objective
            f(w) = || (A @ Dpinv @ C) @ w - (A @ Dpinv @ e - b) ||_2^2
        with minimizer:

        Note: upper_soln_matrix = A @ Dpinv @ C, defined in self.__init__()
        """
        return lstsq(self.upper_soln_matrix, self.A @ (self.Dpinv @ self.e) - self.b)

    def upper_grad_soln(self, w: np.ndarray):
        """
        The true upper-level gradient w.r.t. w, used for validation only
            f'(w) = 2 * (A @ Dpinv @ C).T @ (A @ Dpinv @ C @ w - A @ Dpinv @ e + b) by direct computation
                  = 2 * (A @ Dpinv @ C).T @ (A @ (Dpinv @ (C @ w - e)) + b)
                  = 2 * (A @ Dpinv @ C).T @ (b - A @ x(w))

        Note: upper_soln_matrix = A @ Dpinv @ C, defined in self.__init__()
        """
        assert w.shape == (self.nw,), "w has wrong shape (got %s, expect (%g,))" % (w.shape, self.nw)
        # return 2 * upper_soln_matrix.T @ (upper_soln_matrix @ w - self.A @ (self.Dpinv @ self.e) + self.b)
        x_w = self.lower_soln(w)
        return 2 * self.upper_soln_matrix.T @ (self.b - self.A @ x_w)


def check_derivatives():
    """
    Do some finite difference checks to confirm QuadraticProblem derivative definitions are valid
    """
    # Sample problem and locations where to evaluate derivatives
    nx, nw = 5, 6
    prob = QuadraticProblem(m1=20, m2=21, nx=nx, nw=nw)  # not too large
    x_test = np.ones((prob.nx,))
    w_test = 0.5 * np.ones((prob.nw,))
    v_test = -0.3 * np.ones((prob.nx,))
    hs = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    def e(i,n):
        # i-th coordinate vector
        ei = np.zeros((n,))
        ei[i] = 1.0
        return ei

    print("*** Lower grad params w ***")
    grad = prob.lower_grad_params(x_test, w_test)
    for h in hs:
        fd_grad = np.zeros(w_test.shape)
        for i in range(len(fd_grad)):
            fd_grad[i] = (prob.lower_obj(x_test, w_test + h*e(i,nw)) - prob.lower_obj(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_grad - grad):.2e}")

    print("*** Lower grad inputs x ***")
    grad = prob.lower_grad_inputs(x_test, w_test)
    for h in hs:
        fd_grad = np.zeros(x_test.shape)
        for i in range(len(fd_grad)):
            fd_grad[i] = (prob.lower_obj(x_test + h*e(i,nx), w_test) - prob.lower_obj(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_grad - grad):.2e}")

    print("*** Lower Hess-vec inputs x ***")
    Hv = prob.lower_hess_vec_inputs(x_test, w_test, v_test)
    for h in hs:
        fd_Hv = (prob.lower_grad_inputs(x_test + h*v_test, w_test) - prob.lower_grad_inputs(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_Hv - Hv):.2e}")

    print("*** Lower Hess inputs x ***")
    H = prob.lower_hess_inputs(x_test, w_test)
    for h in hs:
        fd_H = np.zeros((len(x_test), len(x_test)))
        for i in range(fd_H.shape[0]):
            fd_H[i,:] = (prob.lower_grad_inputs(x_test + h * e(i, nx), w_test) - prob.lower_grad_inputs(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_H - H):.2e}")

    print("*** Lower Jac-vec ***")
    vJ = prob.lower_jac_vec(x_test, w_test, v_test)
    for h in hs:
        fd_vJ = (prob.lower_grad_params(x_test + h * v_test, w_test) - prob.lower_grad_params(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_vJ - vJ):.2e}")

    print("*** Lower Jac ***")
    J = prob.lower_jac(x_test, w_test)  # size |x|*|w|
    for h in hs:
        fd_J = np.zeros((len(x_test), len(w_test)))
        for i in range(fd_J.shape[0]):
            fd_J[i, :] = (prob.lower_grad_params(x_test + h * e(i, nx), w_test) - prob.lower_grad_params(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_J - J):.2e}")

    print("*** Upper grad params w ***")
    grad = prob.upper_grad_params(x_test, w_test)
    for h in hs:
        fd_grad = np.zeros(w_test.shape)
        for i in range(len(fd_grad)):
            fd_grad[i] = (prob.upper_obj(x_test, w_test + h * e(i,nw)) - prob.upper_obj(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_grad - grad):.2e}")

    print("*** Upper grad inputs x ***")
    grad = prob.upper_grad_inputs(x_test, w_test)
    for h in hs:
        fd_grad = np.zeros(x_test.shape)
        for i in range(len(fd_grad)):
            fd_grad[i] = (prob.upper_obj(x_test + h * e(i,nx), w_test) - prob.upper_obj(x_test, w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_grad - grad):.2e}")

    print("*** Lower soln grad ***")
    G = prob.lower_soln_grad_params(w_test)  # size |x|*|w|
    for h in hs:
        fd_G = np.zeros((len(x_test), len(w_test)))
        for i in range(fd_G.shape[1]):
            fd_G[:, i] = (prob.lower_soln(w_test + h * e(i, nw)) - prob.lower_soln(w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_G - G):.2e}")

    print("*** Upper full grad soln ***")
    grad = prob.upper_grad_soln(w_test)
    for h in hs:
        fd_grad = np.zeros(w_test.shape)
        for i in range(len(fd_grad)):
            fd_grad[i] = (prob.upper_obj(prob.lower_soln(w_test + h * e(i, nw)), w_test + h * e(i, nw))
                          - prob.upper_obj(prob.lower_soln(w_test), w_test)) / h
        print(f"{h:.2e} {np.linalg.norm(fd_grad - grad):.2e}")

    print("*** Lower soln grad (alt) ***")
    G = prob.lower_soln_grad_params(w_test)  # size |x|*|w|
    x_w = prob.lower_soln(w_test)
    G2 = -np.linalg.solve(prob.lower_hess_inputs(x_w, w_test), prob.lower_jac(x_w, w_test))
    print("Alt method error = %.2e" % np.linalg.norm(G - G2))

    print("*** Upper full grad soln (alt) ***")
    grad = prob.upper_grad_soln(w_test)
    x_w = prob.lower_soln(w_test)
    grad2 = -prob.lower_jac(x_w, w_test).T @ np.linalg.solve(prob.lower_hess_inputs(x_w, w_test), prob.upper_grad_inputs(x_w, w_test)) \
            + prob.upper_grad_params(x_w, w_test)
    print("Alt method error = %.2e" % np.linalg.norm(grad - grad2))
    return


def fista(grad_fn, Y0, L, mu, niters, Ytrue=None):
    """
    This is a modified FISTA for strongly-convex objectives; see Algorithm 1 of
        L. Calatroni and A. Chambolle (2019), Backtracking Strategies for Accelerated Descent Methods
        with Smooth Composite Objectives, SIAM J. Optim. 29(3), pp. 1772-1798.
    or Algorithm 5 of
        A. Chambolle and T. Pock (2016), An introduction to continuous optimization for imaging,
        Acta Numerica 25, pp. 161-319.
    """
    # If Ytrue is not None, measure error to Ytrue at each iteration
    step_size = 1.0 / L
    q = step_size * mu

    # Main loop
    tk = 0.0  # initial value t0
    Yk = Y0.copy()
    Ykm1 = Yk.copy()  # Y_{-1}=Y0 initially

    # Error history
    Yerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Ytrue is not None:
        """
        A priori bound from Theorem 4.10 of Chambolle & Pock (with tau=1/L, mu_g=0)
        
        Converting to a bound on ||xk-x*|| using
            ||xk-x*||^2 <= (2/mu) * [ F(xk) - F(x*) ]
        """
        Y0err = np.linalg.norm(Yk - Ytrue)
        Yerr['true'].append(Y0err)
        Ysq_err = min((1 + sqrt(q)) * (1 - sqrt(q)) ** 0, 4 / (0 + 1) ** 2) * (L / mu) * Y0err ** 2
        Yerr['apriori'].append(sqrt(Ysq_err))
        Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)

    for k in range(niters):
        tksq = tk ** 2
        tkp1 = (1 - q * tksq + sqrt((1 - q * tksq) ** 2 + 4 * tksq)) / 2
        beta_k = (tk - 1) * (1 - q * tkp1) / (tkp1 * (1 - q))
        Zk = Yk + beta_k * (Yk - Ykm1)
        Zk = Zk - step_size * grad_fn(Zk)
        # Update Yk, Ykm1
        Ykm1 = Yk.copy()
        Yk = Zk.copy()
        # Update t
        tk = tkp1
        if Ytrue is not None:
            Yerr['true'].append(np.linalg.norm(Yk - Ytrue))
            Ysq_err = min((1 + sqrt(q)) * (1 - sqrt(q)) ** (k+1), 4 / (k + 2) ** 2) * (L / mu) * Yerr['true'][0] ** 2
            Yerr['apriori'].append(sqrt(Ysq_err))
            Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)

    if Ytrue is not None:
        for key in Yerr:
            Yerr[key] = np.array(Yerr[key])
        return Yk, Yerr
    else:
        return Yk


def gradient_descent(grad_fn, Y0, L, mu, niters, Ytrue=None):
    # Mehmood & Ochs (2020), Remark 7 - convergence rate is qGD
    # If Ytrue is not None, measure errors to Ytrue at each iteration
    qGD = (L - mu) / (L + mu)
    step_size = 2.0 / (L + mu)
    Yk = Y0.copy()

    # Error history
    Yerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Ytrue is not None:
        Y0err = np.linalg.norm(Yk - Ytrue)
        Yerr['true'].append(Y0err)
        Yerr['apriori'].append(Y0err)
        Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)

    for k in range(niters):
        Yk = Yk - step_size * grad_fn(Yk)
        if Ytrue is not None:
            Yerr['true'].append(np.linalg.norm(Yk - Ytrue))
            Yerr['apriori'].append(qGD * Yerr['apriori'][-1])
            Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)
    
    if Ytrue is not None:
        for key in Yerr:
            Yerr[key] = np.array(Yerr[key])
        return Yk, Yerr
    else:
        return Yk


def heavy_ball(grad_fn, Y0, L, mu, niters, Ytrue=None):
    # Parameters from Mehmood & Ochs (2020), Remark 14 - convergence rate is qHB
    # If Ytrue is not None, measure error to Ytrue at each iteration
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    heavy_ball_momentum = qHB ** 2  # beta
    step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha

    Yk = Y0.copy()
    Ykm1 = Yk.copy()

    # Error history
    Yerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Ytrue is not None:
        Y0err = np.linalg.norm(Yk - Ytrue)
        Yerr['true'].append(Y0err)
        Yerr['apriori'].append(Y0err)
        Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)

    for k in range(niters):
        Ynew = Yk - step_size * grad_fn(Yk) + heavy_ball_momentum * (Yk - Ykm1)
        Ykm1 = Yk.copy()
        Yk = Ynew.copy()
        if Ytrue is not None:
            Yerr['true'].append(np.linalg.norm(Yk - Ytrue))
            Yerr['apriori'].append(qHB * Yerr['apriori'][-1])
            Yerr['aposteriori'].append(np.linalg.norm(grad_fn(Yk)) / mu)
    
    if Ytrue is not None:
        for key in Yerr:
            Yerr[key] = np.array(Yerr[key])
        return Yk, Yerr
    else:
        return Yk


def inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, g0, mu, L, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    step_size = 2.0 / (L + mu)

    xbar_inexact = gradf.copy()
    upper_grad = g0.copy()

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(0))

    for k in range(niters):
        upper_grad = upper_grad - step_size * jac_vec_prod(xbar_inexact)
        xbar_inexact = xbar_inexact - step_size * hess_vec_prod(xbar_inexact)
        if Gtrue is not None:
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(k+1))

    if Gtrue is not None:
        for key in Gerr:
            Gerr[key] = np.array(Gerr[key])
        return upper_grad, Gerr
    else:
        return upper_grad


def inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, g0, mu, L, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))
    heavy_ball_momentum = qHB ** 2  # beta
    step_size = 4.0 / (sqrt(L) + sqrt(mu)) ** 2  # alpha

    xbar_inexact = gradf.copy()
    xbar_inexact_prev = np.zeros(xbar_inexact.shape)
    upper_grad = g0.copy()

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(0))

    for k in range(niters):
        upper_grad = upper_grad - step_size * jac_vec_prod(xbar_inexact)
        xbar_inexact_new = xbar_inexact - step_size * hess_vec_prod(xbar_inexact) \
                           + heavy_ball_momentum * (xbar_inexact - xbar_inexact_prev)
        xbar_inexact_prev = xbar_inexact.copy()
        xbar_inexact = xbar_inexact_new
        if Gtrue is not None:
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(k+1))

    if Gtrue is not None:
        for key in Gerr:
            Gerr[key] = np.array(Gerr[key])
        return upper_grad, Gerr
    else:
        return upper_grad


def inexact_ad_ift(gradf, hess_vec_prod, jac_vec_prod, g0, mu, L, niters, Gtrue=None, apriori_est=None, aposteriori_est=None):
    # Use CG to solve the system: hess_vec_prod(q) = gradf <-> Ax=b, starting from g0
    q = g0.copy()
    r = gradf - hess_vec_prod(q)
    d = r.copy()

    # Error history
    Gerr = {'true': [], 'apriori': [], 'aposteriori': []}
    if Gtrue is not None:
        upper_grad = -jac_vec_prod(q)
        G0err = np.linalg.norm(upper_grad - Gtrue)
        Gerr['true'].append(G0err)
        Gerr['apriori'].append(apriori_est(0))
        Gerr['aposteriori'].append(aposteriori_est(np.linalg.norm(q), np.linalg.norm(r)))

    for k in range(niters):
        if np.linalg.norm(r) > 0.0:  # don't run CG while residual is too small
            A_times_d = hess_vec_prod(d)
            alpha = np.dot(r, r) / np.dot(d, A_times_d)
            q = q + alpha * d
            rnew = r - alpha * A_times_d
            beta = np.dot(rnew, rnew) / np.dot(r, r)
            d = rnew + beta * d
            r = rnew.copy()
        # otherwise (||r|| small), do nothing

        if Gtrue is not None:
            upper_grad = -jac_vec_prod(q)
            Gerr['true'].append(np.linalg.norm(upper_grad - Gtrue))
            Gerr['apriori'].append(apriori_est(k+1))
            Gerr['aposteriori'].append(aposteriori_est(np.linalg.norm(q), np.linalg.norm(r)))

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
    plt.semilogy(fista_errs['true'], 'k-', label='FISTA', linewidth=2)
    plt.semilogy(fista_errs['apriori'], 'k--', label='FISTA bound', linewidth=2)
    plt.semilogy(gd_errs['true'], 'b-', label='GD', linewidth=2)
    plt.semilogy(gd_errs['apriori'], 'b--', label='GD bound', linewidth=2)
    plt.semilogy(hb_errs['true'], 'r-', label='HB', linewidth=2)
    plt.semilogy(hb_errs['apriori'], 'r--', label='HB bound', linewidth=2)
    ax = plt.gca()
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Iterate Error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='lower left', fontsize=font_size, fancybox=True)
    plt.grid()
    # plt.legend(loc='best')
    # plt.xlabel('Iterations')
    # plt.ylabel('$||x_k-x(w)||$')
    # plt.title('A priori (from linear rate)')

    plt.savefig('quadratic_problem_plots/compare_lower_level_apriori.%s' % fmt, bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.semilogy(fista_errs['true'], 'k-', label='FISTA', linewidth=2)
    plt.semilogy(fista_errs['aposteriori'], 'k--', label='FISTA bound', linewidth=2)
    plt.semilogy(gd_errs['true'], 'b-', label='GD', linewidth=2)
    plt.semilogy(gd_errs['aposteriori'], 'b--', label='GD bound', linewidth=2)
    plt.semilogy(hb_errs['true'], 'r-', label='HB', linewidth=2)
    plt.semilogy(hb_errs['aposteriori'], 'r--', label='HB bound', linewidth=2)

    ax = plt.gca()
    ax.set_xlabel("Iterations", fontsize=font_size)
    ax.set_ylabel("Iterate Error", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='lower left', fontsize=font_size, fancybox=True)
    plt.grid()
    # plt.legend(loc='best')
    # plt.xlabel('Iterations')
    # plt.ylabel('$||x_k-x(w)||$')
    # plt.title('A posteriori (from gradient)')

    plt.savefig('quadratic_problem_plots/compare_lower_level_aposteriori.%s' % fmt, bbox_inches='tight')
    plt.show()
    return


def compare_inexact_ad(font_size='large', fmt='png'):
    prob = QuadraticProblem(m1=1000, m2=1000, nx=10, nw=10)
    w = np.ones((prob.nw,))
    x_w = prob.lower_soln(w) + 1e-15 * np.ones((prob.nx,))  # TODO use true lower-level solution x(w) for now
    L, mu = prob.lip_const(w), prob.convex_const(w)
    # x_w = heavy_ball(lambda x: prob.lower_grad_inputs(x, w), np.zeros((prob.nx,)), L, mu, 50)
    gradf = prob.upper_grad_inputs(x_w, w)
    hess_vec_prod = lambda v: prob.lower_hess_vec_inputs(x_w, w, v)
    jac_vec_prod = lambda v: prob.lower_jac_vec(x_w, w, v)
    g0 = np.zeros((prob.nw,))
    niters = 200
    Gtrue = -prob.lower_jac(x_w, w).T @ np.linalg.solve(prob.lower_hess_inputs(x_w, w), gradf)  # full gradient using above x(w) value
    # Gtrue = prob.upper_grad_soln(w)  # full gradient using exact x(w)

    # A priori estimators
    Bmax = prob.jac_bound()
    Lf = prob.lip_upper()
    LB = prob.lip_jac()
    LA = prob.lip_hess()
    x_w_true = prob.lower_soln(w)
    eps = np.linalg.norm(x_w - x_w_true)
    print("x(w) error = %g" % eps)
    norm_gradf_xtrue = np.linalg.norm(prob.upper_grad_inputs(x_w_true, w))
    D = norm_gradf_xtrue * LB / mu + Bmax * norm_gradf_xtrue * LA / (mu ** 2) + Bmax * Lf / mu
    qGD = (L - mu) / (L + mu)
    qHB = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))

    def apriori_est_ift(k):
        return 2 * (L / mu) ** 1.5 * (qHB ** k) * (norm_gradf_xtrue + Lf * eps) + D * eps

    def apriori_est_gd(k):
        return (qGD ** k) * Bmax * (norm_gradf_xtrue + Lf * eps) / mu + D * eps

    def apriori_est_hb(k):
        # Assuming C=1 and gamma=0 (but we can't verify this
        return (qHB ** k) * Bmax * (norm_gradf_xtrue + Lf * eps) / mu + D * eps

    Geps = np.linalg.norm(prob.lower_grad_inputs(x_w, w)) / mu  # approx 5.6e-15 for true x_w
    Bx_norm = np.linalg.norm(prob.lower_jac(x_w, w), ord=2)

    def aposteriori_est_ift(qnorm, rnorm):  # depends on norms of current iterate ||q||_2 and residual ||r||_2
        # Main note: not using neighborhood Lipschitz constants
        return (Bx_norm / mu) * rnorm \
               + ((Bx_norm * Lf + Bx_norm * qnorm * LA) / mu + qnorm * LB) * Geps \
               + LB * Geps * (rnorm / mu + Geps * (Lf + qnorm * LA) / mu)

    norm_gradf_x_w = np.linalg.norm(prob.upper_grad_inputs(x_w, w))
    Deps = norm_gradf_x_w * LB / mu + Bmax * norm_gradf_x_w * LA / (mu ** 2) + Bmax * Lf / mu

    def aposteriori_est_gd(k):
        return qGD ** k * Bmax * norm_gradf_x_w / mu + Deps * eps

    def aposteriori_est_hb(k):
        return qHB ** k * Bmax * norm_gradf_x_w / mu + Deps * eps

    _, ift_errs = inexact_ad_ift(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=Gtrue,
                                 apriori_est=apriori_est_ift, aposteriori_est=aposteriori_est_ift)
    _, gd_errs = inexact_ad_gd(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=Gtrue,
                               apriori_est=apriori_est_gd, aposteriori_est=aposteriori_est_gd)
    _, hb_errs = inexact_ad_hb(gradf, hess_vec_prod, jac_vec_prod, g0, L, mu, niters, Gtrue=Gtrue,
                               apriori_est=apriori_est_hb, aposteriori_est=aposteriori_est_hb)

    plt.figure(1)
    plt.clf()
    plt.semilogy(ift_errs['true'], 'k-', label='IFT', linewidth=2)
    plt.semilogy(ift_errs['apriori'], 'k--', label='IFT bound', linewidth=2)
    plt.semilogy(gd_errs['true'], 'b-', label='GD', linewidth=2)
    plt.semilogy(gd_errs['apriori'], 'b--', label='GD bound', linewidth=2)
    plt.semilogy(hb_errs['true'], 'r-', label='HB', linewidth=2)
    plt.semilogy(hb_errs['apriori'], 'r--', label='HB bound', linewidth=2)
    ax = plt.gca()
    ax.set_xlabel("AD Iterations", fontsize=font_size)
    ax.set_ylabel('Gradient Error', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size, fancybox=True)
    plt.grid()
    # plt.legend(loc='best')
    # plt.xlabel('AD Iterations')
    # plt.ylabel('Gradient Error')
    # plt.title('A priori')

    plt.savefig('quadratic_problem_plots/compare_inexact_ad_apriori.%s' % fmt, bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.semilogy(ift_errs['true'], 'k-', label='IFT', linewidth=2)
    plt.semilogy(ift_errs['aposteriori'], 'k--', label='IFT bound', linewidth=2)
    plt.semilogy(gd_errs['true'], 'b-', label='GD', linewidth=2)
    plt.semilogy(gd_errs['aposteriori'], 'b--', label='GD bound', linewidth=2)
    plt.semilogy(hb_errs['true'], 'r-', label='HB', linewidth=2)
    plt.semilogy(hb_errs['aposteriori'], 'r--', label='HB bound', linewidth=2)
    ax = plt.gca()
    ax.set_xlabel("AD Iterations", fontsize=font_size)
    ax.set_ylabel('Gradient Error', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size, fancybox=True)
    plt.grid()
    # plt.legend(loc='best')
    # plt.xlabel('AD Iterations')
    # plt.ylabel('Gradient Error')
    # plt.title('A posteriori')

    plt.savefig('quadratic_problem_plots/compare_inexact_ad_aposteriori.%s' % fmt, bbox_inches='tight')
    plt.show()
    return


def main():
    run_deriv_checks = False
    if run_deriv_checks:
        print("Checking derivatives...")
        print("")
        check_derivatives()
        print("Done")
        return
    # (end derivative checks)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font_size = 'x-large'
    fmt = 'pdf'

    compare_lower_level_solvers(font_size=font_size, fmt=fmt)
    compare_inexact_ad(font_size=font_size, fmt=fmt)

    print("Done")
    return


if __name__ == '__main__':
    main()