#!/usr/bin/env python3

"""
Reimplementation of ODL's PDHG routine:
https://github.com/odlgroup/odl/blob/master/odl/solvers/nonsmooth/primal_dual_hybrid_gradient.py

Modified to include custom stopping criteria
"""
import numpy as np
import odl

__all__ = ['pdhg']


def pdhg(x, f, g, L, niter, tau=None, sigma=None, **kwargs):
    r"""Primal-dual hybrid gradient algorithm for convex optimization.
    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is ::
        min_{x in X} f(x) + g(L x)
    where ``L`` is an operator and ``f`` and ``g`` are functionals.
    The primal-dual hybrid-gradient algorithm is a primal-dual algorithm, and
    basically consists of alternating a gradient ascent in the dual variable
    and a gradient descent in the primal variable. The proximal operator is
    used to generate a ascent direction for the convex conjugate of F and
    descent direction for G. Additionally an over-relaxation of the primal
    variable is performed.
    Parameters
    ----------
    x : ``L.domain`` element
        Starting point of the iteration, updated in-place.
    f : `Functional`
        The function ``f`` in the problem definition. Needs to have
        ``f.proximal``.
    g : `Functional`
        The function ``g`` in the problem definition. Needs to have
        ``g.convex_conj.proximal``.
    L : linear `Operator`
        The linear operator that should be applied before ``g``. Its range must
        match the domain of ``g`` and its domain must match the domain of
        ``f``.
    niter : non-negative int
        Number of iterations.
    tau : float, optional
        Step size parameter for ``g``.
        Default: Sufficient for convergence, see `pdhg_stepsize`.
    sigma : sequence of floats, optional
        Step size parameters for ``f``.
        Default: Sufficient for convergence, see `pdhg_stepsize`.
    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    theta : float, optional
        Relaxation parameter, required to fulfill ``0 <= theta <= 1``.
        Default: 1
    gamma_primal : non-negative float, optional
        Acceleration parameter. If not ``None``, it overrides ``theta`` and
        causes variable relaxation parameter and step sizes to be used,
        with ``tau`` and ``sigma`` as initial values. Requires ``f`` to be
        strongly convex and ``gamma_primal`` being upper bounded by the strong
        convexity constant of ``f``. Acceleration can either be done on the
        primal part or the dual part but not on both simultaneously.
        Default: ``None``
    gamma_dual : non-negative float, optional
        Acceleration parameter as ``gamma_primal`` but for dual variable.
        Requires ``g^*`` to be strongly convex and ``gamma_dual`` being upper
        bounded by the strong convexity constant of ``f^*``. Acceleration can
        either be done on the primal part or the dual part but not on both
        simultaneously.
        Default: ``None``
    x_relax : ``op.domain`` element, optional
        Required to resume iteration. For ``None``, a copy of the primal
        variable ``x`` is used.
        Default: ``None``
    y : ``op.range`` element, optional
        Required to resume iteration. For ``None``, ``op.range.zero()``
        is used.
        Default: ``None``
    tol : positive float, optional
        If not None, terminate early if the objective value at a new point is at most
         ``tol`` less than the best objective value seen so far. Do not terminate on
         an iteration which does not improve on the objective value.
        Default: ``None``
    Notes
    -----
    The problem of interest is
    .. math::
        \min_{x \in X} f(x) + g(L x),
    where the formal conditions are that :math:`L` is an operator
    between Hilbert spaces :math:`X` and :math:`Y`.
    Further, :math:`f : X \rightarrow [0, +\infty]` and
    :math:`g : Y \rightarrow [0, +\infty]` are proper, convex,
    lower-semicontinuous functionals.
    Convergence is only guaranteed if :math:`L` is linear, :math:`X, Y`
    are finite dimensional and the step lengths :math:`\sigma` and
    :math:`\tau` satisfy
    .. math::
       \tau \sigma \|L\|^2 < 1
    where :math:`\|L\|` is the operator norm of :math:`L`.
    It is often of interest to study problems that involve several operators,
    for example the classical TV regularized problem
    .. math::
        \min_x \|Ax - b\|_2^2 + \|\nabla x\|_1.
    Here it is tempting to let :math:`f(x)=\|\nabla x\|_1`, :math:`L=A` and
    :math:`g(y)=||y||_2^2`. This is however not feasible since the
    proximal of :math:`||\nabla x||_1` has no closed form expression.
    Instead, the problem can be formulated :math:`f(x)=0`,
    :math:`L(x) = (A(x), \nabla x)` and
    :math:`g((x_1, x_2)) = \|x_1\|_2^2 + \|x_2\|_1`. See the
    examples folder for more information on how to do this.
    For a more detailed documentation see `the PDHG guide
    <https://odlgroup.github.io/odl/guide/pdhg_guide.html>`_ in the online
    documentation.
    References on the algorithm can be found in `[CP2011a]
    <https://doi.org/10.1007/s10851-010-0251-1>`_ and `[CP2011b]
    <https://doi.org/10.1109/ICCV.2011.6126441>`_.
    This implementation of the CP algorithm is along the lines of
    `[Sid+2012] <https://doi.org/10.1088/0031-9155/57/10/3065>`_.
    The non-linear case is analyzed in `[Val2014]
    <https://doi.org/10.1088/0266-5611/30/5/055012>`_.
    See Also
    --------
    odl.solvers.nonsmooth.douglas_rachford.douglas_rachford_pd :
        Solver for similar problems which can additionaly handle infimal
        convolutions and multiple forward operators.
    odl.solvers.nonsmooth.forward_backward.forward_backward_pd :
        Solver for similar problems which can additionaly handle infimal
        convolutions, multiple forward operators and a differentiable term.
    References
    ----------
    [CP2011a] Chambolle, A and Pock, T. *A First-Order
    Primal-Dual Algorithm for Convex Problems with Applications to
    Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011),
    pp 120-145.
    [CP2011b] Chambolle, A and Pock, T. *Diagonal
    preconditioning for first order primal-dual algorithms in convex
    optimization*. 2011 IEEE International Conference on Computer Vision
    (ICCV), 2011, pp 1762-1769.
    [Sid+2012] Sidky, E Y, Jorgensen, J H, and Pan, X.
    *Convex optimization problem prototyping for image reconstruction in
    computed tomography with the Chambolle-Pock algorithm*. Physics in
    Medicine and Biology, 57 (2012), pp 3065-3091.
    [Val2014] Valkonen, T.
    *A primal-dual hybrid gradient method for non-linear operators with
    applications to MRI*. Inverse Problems, 30 (2014).
    """
    # Forward operator
    if not isinstance(L, odl.Operator):
        raise TypeError('`op` {!r} is not an `Operator` instance'
                        ''.format(L))

    # Starting point
    if x not in L.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, L.domain))

    # Spaces
    if f.domain != L.domain:
        raise TypeError('`f.domain` {!r} must equal `op.domain` {!r}'
                        ''.format(f.domain, L.domain))

    # Step size parameters
    tau, sigma = pdhg_stepsize(L, tau, sigma)

    # Number of iterations
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('`niter` {} not understood'
                         ''.format(niter))

    # Relaxation parameter
    theta = kwargs.pop('theta', 1)
    theta, theta_in = float(theta), theta
    if not 0 <= theta <= 1:
        raise ValueError('`theta` {} not in [0, 1]'
                         ''.format(theta_in))

    # Acceleration parameters
    gamma_primal = kwargs.pop('gamma_primal', None)
    if gamma_primal is not None:
        gamma_primal, gamma_primal_in = float(gamma_primal), gamma_primal
        if gamma_primal < 0:
            raise ValueError('`gamma_primal` must be non-negative, got {}'
                             ''.format(gamma_primal_in))

    gamma_dual = kwargs.pop('gamma_dual', None)
    if gamma_dual is not None:
        gamma_dual, gamma_dual_in = float(gamma_dual), gamma_dual
        if gamma_dual < 0:
            raise ValueError('`gamma_dual` must be non-negative, got {}'
                             ''.format(gamma_dual_in))

    if gamma_primal is not None and gamma_dual is not None:
        raise ValueError('Only one acceleration parameter can be used')

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Initialize the relaxation variable
    x_relax = kwargs.pop('x_relax', None)
    if x_relax is None:
        x_relax = x.copy()
    elif x_relax not in L.domain:
        raise TypeError('`x_relax` {} is not in the domain of '
                        '`L` {}'.format(x_relax.space, L.domain))

    # Initialize the dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = L.range.zero()
    elif y not in L.range:
        raise TypeError('`y` {} is not in the range of `L` '
                        '{}'.format(y.space, L.range))

    # Initialize the termination tolerance
    tol = kwargs.pop('tol', None)
    if tol is not None:
        tol, tol_in = float(tol), tol
        if tol <= 0:
            raise ValueError('`tol` must be non-negative, got {}'
                             ''.format(tol_in))

    # Get the proximals
    proximal_primal = f.proximal
    proximal_dual = g.convex_conj.proximal
    proximal_constant = (gamma_primal is None) and (gamma_dual is None)
    if proximal_constant:
        # Pre-compute proximals for efficiency
        proximal_dual_sigma = proximal_dual(sigma)
        proximal_primal_tau = proximal_primal(tau)

    # Temporary copy to store previous iterate and best objective value so far
    x_old = x.space.element()
    obj_best = None
    if tol is not None:
        obj_best = float(f(x) + g(L(x)))

    # Temporaries
    dual_tmp = L.range.element()
    primal_tmp = L.domain.element()

    for _ in range(niter):
        # Copy required for relaxation
        x_old.assign(x)

        # Gradient ascent in the dual variable y
        # Compute dual_tmp = y + sigma * L(x_relax)
        L(x_relax, out=dual_tmp)
        dual_tmp.lincomb(1, y, sigma, dual_tmp)

        # Apply the dual proximal
        if not proximal_constant:
            proximal_dual_sigma = proximal_dual(sigma)
        proximal_dual_sigma(dual_tmp, out=y)

        # Gradient descent in the primal variable x
        # Compute primal_tmp = x + (- tau) * L.derivative(x).adjoint(y)
        L.derivative(x).adjoint(y, out=primal_tmp)
        primal_tmp.lincomb(1, x, -tau, primal_tmp)

        # Apply the primal proximal
        if not proximal_constant:
            proximal_primal_tau = proximal_primal(tau)
        proximal_primal_tau(primal_tmp, out=x)

        # Acceleration
        if gamma_primal is not None:
            theta = float(1 / np.sqrt(1 + 2 * gamma_primal * tau))
            tau *= theta
            sigma /= theta

        if gamma_dual is not None:
            theta = float(1 / np.sqrt(1 + 2 * gamma_dual * sigma))
            tau /= theta
            sigma *= theta

        # Over-relaxation in the primal variable x
        x_relax.lincomb(1 + theta, x, -theta, x_old)

        if callback is not None:
            callback(x)

        # Terminate on slow objective decrease
        if tol is not None:
            obj = float(f(x) + g(L(x)))
            if obj_best - tol <= obj <= obj_best:
                break
            obj_best = min(obj_best, obj)
    return


def pdhg_stepsize(L, tau=None, sigma=None):
    r"""Default step sizes for `pdhg`.
    Parameters
    ----------
    L : `Operator` or float
        Operator or norm of the operator that are used in the `pdhg` method.
        If it is an `Operator`, the norm is computed with
        ``Operator.norm(estimate=True)``.
    tau : positive float, optional
        Use this value for ``tau`` instead of computing it from the
        operator norms, see Notes.
    sigma : positive float, optional
        The ``sigma`` step size parameters for the dual update.
    Returns
    -------
    tau : float
        The ``tau`` step size parameter for the primal update.
    sigma : tuple of float
        The ``sigma`` step size parameter for the dual update.
    Notes
    -----
    To guarantee convergence, the parameters :math:`\tau`, :math:`\sigma`
    and :math:`L` need to satisfy
    .. math::
       \tau \sigma \|L\|^2 < 1
    This function has 4 options, :math:`\tau`/:math:`\sigma` given or not
    given.
    - Neither :math:`\tau` nor :math:`\sigma` are given, they are chosen as
      .. math::
          \tau = \sigma = \frac{\sqrt{0.9}}{\|L\|}
    - If only :math:`\sigma` is given, :math:`\tau` is set to
      .. math::
          \tau = \frac{0.9}{\sigma \|L\|^2}
    - If only :math:`\tau` is given, :math:`\sigma` is set
      to
      .. math::
          \sigma = \frac{0.9}{\tau \|L\|^2}
    - If both are given, they are returned as-is without further validation.
    """
    if tau is not None and sigma is not None:
        return float(tau), float(sigma)

    L_norm = L.norm(estimate=True) if isinstance(L, odl.Operator) else float(L)
    if tau is None and sigma is None:
        tau = sigma = np.sqrt(0.9) / L_norm
        return tau, sigma
    elif tau is None:
        tau = 0.9 / (sigma * L_norm ** 2)
        return tau, float(sigma)
    else:  # sigma is None
        sigma = 0.9 / (tau * L_norm ** 2)
        return float(tau), sigma


def main():
    # Basic TV denoising example from
    # https://odlgroup.github.io/odl/guide/pdhg_guide.html
    space = odl.uniform_discr(min_pt=[0, 0], max_pt=[100, 100], shape=[256, 256])
    phantom = odl.phantom.shepp_logan(space, modified=True)
    data = phantom + odl.phantom.white_noise(space) * 0.1
    ident = odl.IdentityOperator(space)
    grad = odl.Gradient(space)
    L = odl.BroadcastOperator(ident, grad)
    l2_norm_squared = odl.solvers.L2NormSquared(space).translated(data)
    l1_norm = 0.0003 * odl.solvers.L1Norm(grad.range)
    g = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)
    f = odl.solvers.IndicatorNonnegativity(space)
    op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=4, xstart=phantom)
    tau = sigma = 1.0 / op_norm
    x = space.zero()

    class BasicCallback(odl.solvers.Callback):

        def __init__(self):
            self.iter_count = 0

        def __call__(self, x, **kwargs):
            print("At iter %g" % self.iter_count)
            self.iter_count += 1

    callback = BasicCallback()
    pdhg(x, f, g, L, tau=tau, sigma=sigma, niter=100, callback=callback, tol=1e-2)

    import matplotlib.pyplot as plt
    try:
        from .odl_plotting import plot
    except ModuleNotFoundError:
        from odl_plotting import plot
    plt.figure()
    plt.subplot(1, 3, 1)
    plot(phantom)
    plt.title('phantom')
    plt.subplot(1, 3, 2)
    plot(data)
    plt.title('noisy data')
    plt.subplot(1, 3, 3)
    plot(x)
    plt.title('TV denoised result')
    plt.show()
    return


if __name__ == '__main__':
    main()
