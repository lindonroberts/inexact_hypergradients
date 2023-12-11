"""
A collection of standard optimization routines
"""
from datetime import datetime
from math import ceil, log, sqrt
import torch

__all__ = ['estimate_fista_maxiters', 'fista', 'conjugate_gradient']


def estimate_fista_maxiters(model, Y0, niters=None, y0tol=None, ytol=None):
    # Solve this lower-level problem with FISTA
    # Terminate after niters or ||Yk-Y*|| <= ytol
    # y0tol is initial estimate ||Y0-Y*|| <= y0tol, if available (estimated otherwise)
    norm = lambda v: float(torch.linalg.vector_norm(v, ord=2))
    L, mu = model.lip_const(), model.convex_const()
    dist_to_soln = lambda grad_at_Y: norm(grad_at_Y) / mu  # upper bound on ||Y-Y*|| given self.grad(Y)
    step_size = 1.0 / L
    q = step_size * mu
    w = 1.0 - sqrt(q)

    if niters is not None:
        # Run for fixed number of iterations
        max_iters = niters
    else:
        # Terminate when ||Yk-Y*|| <= ytol
        if y0tol is None:
            y0tol = dist_to_soln(model.gradient_wrt_inputs(Y0))

        max_iters = max(ceil(log(ytol ** 2 / ((L / mu) * (1 + sqrt(q)) * y0tol ** 2), w)), 0)
    return max_iters


def fista(model, Y0, niters=None, y0tol=None, ytol=None, verbose=False, max_wall_runtime_seconds=None, tic=None):
    # Solve this lower-level problem with FISTA
    # Terminate after niters or ||Yk-Y*|| <= ytol, or max timeout
    # y0tol is initial estimate ||Y0-Y*|| <= y0tol, if available (estimated otherwise)

    # Check for timeout
    timeout_exceeded = lambda toc: (toc - tic).total_seconds() > max_wall_runtime_seconds if max_wall_runtime_seconds is not None else False

    copy = lambda v: v.detach().clone()
    norm = lambda v: float(torch.linalg.vector_norm(v, ord=2))
    L, mu = model.lip_const(), model.convex_const()
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
            y0tol = dist_to_soln(model.gradient_wrt_inputs(Y0))

        max_iters = max(ceil(log(ytol**2 / ((L / mu) * (1 + sqrt(q)) * y0tol**2), w)), 0)
        # print("Will terminate lower-level FISTA after at most %g iters" % max_iters)
        terminate_on_small_gradient = True

    Y = copy(Y0)
    terminated_on_small_gradient = False

    # Main loop
    k = 0
    tk = 0.0  # initial value t0
    Ykm1 = copy(Y)  # Y_{-1}=Y0 initially
    while k < max_iters and not timeout_exceeded(datetime.now()):
        tksq = tk ** 2
        tkp1 = (1 - q * tksq + sqrt((1 - q * tksq) ** 2 + 4 * tksq)) / 2
        beta_k = (tk - 1) * (1 - q * tkp1) / (tkp1 * (1 - q))
        Zk = Y + beta_k * (Y - Ykm1)
        Zk.add_(model.gradient_wrt_inputs(Zk), alpha=-step_size)  # Zk = Zk - step_size * self.gradient(Zk)
        # Update Y, Ykm1
        Ykm1 = copy(Y)
        Y = copy(Zk)
        # Update t
        tk = tkp1
        # Check termination
        k += 1
        g = model.gradient_wrt_inputs(Y)
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
    obj = model.objfun(Y)
    grad = model.gradient_wrt_inputs(Y)
    return Y, float(obj), norm(grad), k, terminated_on_small_gradient


def conjugate_gradient(matrix_vector_product, b, x0=None, rtol=1e-10, maxiter=None, verbose=False, save_history=False):
    """
    Run CG until residual tolerance ||Ax-b|| <= rtol achieved. Returns # iterations and true residual ||Ax-b||

    Only access matrix through Ax = matrix_vector_product(x)
    """
    # Set defaults
    if x0 is None:
        x0 = torch.zeros_like(b)
    if maxiter is None:
        maxiter = b.numel()

    # Linear algebra operations in torch (given x and b may not be flattened vectors)
    copy = lambda v: v.detach().clone()  # make copy of a tensor
    dot = lambda v1, v2: (v1 * v2).sum()  # dot product
    norm = lambda v: float(torch.linalg.vector_norm(v.flatten(), ord=2))  # norm

    # Initialization
    x = copy(x0)
    r = b - matrix_vector_product(x)
    d = copy(r)
    k = 0

    if save_history:
        xhist = [copy(x)]
    else:
        xhist = None

    # Main CG loop
    while k < maxiter and norm(r) > rtol:
        if verbose:
            print("CG iter %g/%g: ||r|| = %g, rtol = %g" % (k, maxiter, norm(r), rtol))

        A_times_d = matrix_vector_product(d)
        alpha = dot(r, r) / dot(d, A_times_d)
        x.add_(d, alpha=alpha)  # x = x + alpha*d
        rnew = r - alpha * A_times_d
        beta = dot(rnew, rnew) / dot(r, r)
        d = rnew + beta * d
        r = copy(rnew)
        k += 1
        if save_history:
            xhist.append(copy(x))

    if verbose:
        print("CG finished on iter %g/%g: ||r|| = %g, rtol = %g" % (k, maxiter, norm(r), rtol))
    if save_history:
        return x, k, norm(r), xhist
    else:
        return x, k, norm(r)


def main():
    return


if __name__ == '__main__':
    main()