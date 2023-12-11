"""
Collection of standard regularizers

Implementation of TGV from
https://github.com/odlgroup/odl/blob/master/examples/solvers/pdhg_denoising_tgv.py

Implementation of TV from
https://github.com/mehrhardt/Enhancing_hMRI/blob/main/code/misc.py
"""
import numpy as np
import odl
try:
    from .odl_pdhg import pdhg
except ModuleNotFoundError:
    from odl_pdhg import pdhg

__all__ = ['tgv_denoise', 'tv_denoise']


# Flags for a specific type of image
IMG_1D = 0
IMG_2D_BW = 1
IMG_2D_RGB = 2


def get_img_type(domain):
    """
    Determine if a domain corresponds to 1D, 2D B&W or 2D RGB image
    """
    if type(domain) == odl.ProductSpace:
        assert domain[0].ndim == 2, "For TV/TGV, product space must be based on 2D sub-domain"
        return IMG_2D_RGB
    else:
        assert domain.ndim in [1, 2], "For TV/TGV, domain must be 1D or 2D"
        return IMG_1D if domain.ndim == 1 else IMG_2D_BW


def color_gradient(domain):
    """
    Gradient operator for a color image
    """
    Dx = odl.PartialDerivative(domain[0], 0, method='forward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(domain[0], 1, method='forward', pad_mode='symmetric')
    P0 = odl.ComponentProjection(domain, 0)
    P1 = odl.ComponentProjection(domain, 1)
    P2 = odl.ComponentProjection(domain, 2)
    return odl.BroadcastOperator(Dx * P0, Dy * P0, Dx * P1, Dy * P1, Dx * P2, Dy * P2)


class MyCallback(odl.solvers.Callback):

    def __init__(self):
        self.iter_count = 0

    def __call__(self, x, **kwargs):
        recons, grad = x[0], x[1]
        print(self.iter_count, recons.norm(), grad.norm())  #primal[0].norm(), primal[1].norm())
        self.iter_count += 1


def tgv_denoise(domain, fwd_op, data, alpha=0.1, beta=1.0, niters=400, tol=None, use_custom_pdhg=True, prewhiten=True):
    img_type = get_img_type(domain)

    # Initialize gradient operator
    if img_type == IMG_1D:
        G = odl.PartialDerivative(domain, 0, method='forward', pad_mode='symmetric')
    elif img_type == IMG_2D_BW:
        G = odl.Gradient(domain, method='forward', pad_mode='symmetric')
    elif img_type == IMG_2D_RGB:
        G = color_gradient(domain)
    else:
        raise RuntimeError("Unknown img_type: %g" % img_type)
    V = G.range

    # Create symmetrized operator and weighted space.
    if img_type == IMG_1D:
        E = odl.PartialDerivative(domain, 0, method='backward', pad_mode='symmetric')
    elif img_type == IMG_2D_BW:
        Dx = odl.PartialDerivative(domain, 0, method='backward', pad_mode='symmetric')
        Dy = odl.PartialDerivative(domain, 1, method='backward', pad_mode='symmetric')

        # TODO: As the weighted space is currently not supported in ODL we find a workaround.
        # W = odl.ProductSpace(U, 3, weighting=[1, 1, 2])
        # sym_gradient = odl.operator.ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5*Dy, 0.5*Dx]], range=W)
        E = odl.operator.ProductSpaceOperator([[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
    elif img_type == IMG_2D_RGB:
        Dx = odl.PartialDerivative(domain[0], 0, method='backward', pad_mode='symmetric')
        Dy = odl.PartialDerivative(domain[0], 1, method='backward', pad_mode='symmetric')

        Q0 = odl.ComponentProjection(V, 0)
        Q1 = odl.ComponentProjection(V, 1)
        Q2 = odl.ComponentProjection(V, 2)
        Q3 = odl.ComponentProjection(V, 3)
        Q4 = odl.ComponentProjection(V, 4)
        Q5 = odl.ComponentProjection(V, 5)
        E = odl.operator.BroadcastOperator(
            Dx * Q0, Dy * Q1, (Dy * Q0 + Dx * Q1) / np.sqrt(2),
            Dx * Q2, Dy * Q3, (Dy * Q2 + Dx * Q3) / np.sqrt(2),
            Dx * Q4, Dy * Q5, (Dy * Q4 + Dx * Q5) / np.sqrt(2))
        """
        V = V1, V2
        3 rows of E are 3 color channels. For each channel:
        matrix:
        Dx[V1], avg(Dy[V1], Dx[V2])
        avg(Dy[V1], Dx[V2]), Dy[V2]
        ***Check if avg or not
        Compute Frobenius norm of matrix M = [[A,B],[B.T,C]] from above
        ||M||_F^2 = ||A||_F^2 + 2||B||_F^2 + ||C||_F^2
        = ||vec(A)||_2^2 + 2||vec(B)||_2^2 + ||vec(C)||_2^2
        = ||vec(A)||_2^2 + ||sqrt(2)*vec(B)||_2^2 + ||vec(C)||_2^2
        = ||vec(A, sqrt(2)*B, C)||_2^2
        B = avg(...) so sqrt(2)*B = sum(...)/sqrt(2)
        """
    else:
        raise RuntimeError("Unknown img_type: %g" % img_type)
    W = E.range

    # Create the domain of the problem, given by the reconstruction space and the
    # range of the gradient on the reconstruction space.
    tgv_domain = odl.ProductSpace(domain, V)

    # Column vector of three operators defined as:
    # 1. Computes ``Ax``
    # 2. Computes ``Gx - y``
    # 3. Computes ``Ey``
    A1 = fwd_op * odl.ComponentProjection(tgv_domain, 0)
    A2 = odl.ReductionOperator(G, odl.ScalingOperator(V, -1))
    A3 = E * odl.ComponentProjection(tgv_domain, 1)
    if prewhiten:
        nA1 = A1.norm(estimate=True)  # scale by norm (pre-whitening)
        nA2 = A2.norm(estimate=True)
        nA3 = A3.norm(estimate=True)
    else:
        nA1, nA2, nA3 = 1.0, 1.0, 1.0
    op = odl.BroadcastOperator(
        A1/nA1,
        A2/nA2,
        A3/nA3)

    # Do not use the f functional, set it to zero.
    f = odl.solvers.ZeroFunctional(tgv_domain)

    # l2-squared data matching
    l2_norm = odl.solvers.L2NormSquared(fwd_op.range).translated(data)

    # The l1-norms scaled by regularization parameters
    # ODL demo has odl.solvers.L1Norm, but GroupL1Norm is usually better (but same for 1D but GroupL1Norm doesn't work)
    if img_type == IMG_1D:
        l1_norm_1 = alpha * odl.solvers.L1Norm(V)
        l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)
    else:
        l1_norm_1 = alpha * odl.solvers.GroupL1Norm(V)
        l1_norm_2 = alpha * beta * odl.solvers.GroupL1Norm(W)

    # Combine functionals, order must correspond to the operator K
    g = odl.solvers.SeparableSum(l2_norm*nA1, l1_norm_1*nA2, l1_norm_2*nA3)

    # --- Select solver parameters and solve using PDHG --- #

    # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
    op_norm = 1.1 * odl.power_method_opnorm(op, xstart=op.domain.one())

    gamma = 1.0
    tau = gamma / op_norm  # Step size for the primal variable
    sigma = 1.0 / (gamma * op_norm)  # Step size for the dual variable

    # Choose a starting point
    x = op.domain.zero()
    # x = op.domain.element()
    # x[0][:] = 0.0
    # x[1][:] = 0.0

    # Run the algorithm
    # callback = MyCallback()
    # print("x =", x)  # this changes the result?? (with print = success?)
    # print("f =", f)
    # print("g =", g)
    # print("op =", op)
    # print("niters =", niters)
    # print("tau =", tau)
    # print("sigma =", sigma)
    # return x[0]
    # callback(x)
    # print("x =", x.asarray())
    # callback(x)
    # return x[0]
    # y = op.range.zero()
    if use_custom_pdhg and tol > 0.0:
        pdhg(x, f, g, op, niter=niters, tau=tau, sigma=sigma, tol=tol)
    else:
        odl.solvers.pdhg(x, f, g, op, niter=niters, tau=tau, sigma=sigma)
    return x[0]  # x[1] has derivative info


def tv_denoise(domain, fwd_op, data, alpha=0.1, niters=400, tol=None, use_custom_pdhg=True, start_from_data=False):
    # Hint on specific PDHG formulation is given at the end of this page:
    # https://odlgroup.github.io/odl/generated/odl.solvers.nonsmooth.primal_dual_hybrid_gradient.pdhg.html
    img_type = get_img_type(domain)

    # Initialize gradient operator
    if img_type == IMG_1D or img_type == IMG_2D_BW:
        G = odl.Gradient(domain, method='forward', pad_mode='symmetric')
    else:
        G = color_gradient(domain)

    # Column vector op(x) = [fwd_op(x), grad(x)]
    A1 = fwd_op
    A2 = G
    nA1 = A1.norm(estimate=True)
    nA2 = A2.norm(estimate=True)
    op = odl.BroadcastOperator(A1/nA1, A2/nA2)

    # Do not use the f functional, set it to zero.
    f = odl.solvers.ZeroFunctional(domain)

    # l2-squared data matching
    l2_norm = odl.solvers.L2NormSquared(fwd_op.range).translated(data)

    # The l1-norms scaled by regularization parameters
    l1_norm = alpha * odl.solvers.GroupL1Norm(G.range)

    # Combine functionals, order must correspond to the operator 'op'
    # i.e. g(x) = l2_norm(op1(x)) + l1_norm(op2(x)) = l2_norm(fwd_op(x)) + l1_norm(G(x))
    g = odl.solvers.SeparableSum(l2_norm*nA1, l1_norm*nA2)

    # --- Select solver parameters and solve using PDHG --- #

    # Estimated operator norm, add 10 percent to ensure ||op||_2^2 * sigma * tau < 1
    op_norm = 1.1 * odl.power_method_opnorm(op)

    gamma = 1.0
    tau = gamma / op_norm  # Step size for the primal variable
    sigma = 1.0 / (gamma * op_norm)  # Step size for the dual variable

    # Choose a starting point
    if start_from_data:
        x = data
    else:
        x = op.domain.zero()

    # Run the algorithm
    if use_custom_pdhg and tol > 0.0:
        pdhg(x, f, g, op, niter=niters, tau=tau, sigma=sigma, tol=tol)
    else:
        odl.solvers.pdhg(x, f, g, op, niter=niters, tau=tau, sigma=sigma)
    return x


def run_demo_1d(reg='tgv', niters=400):
    domain = odl.uniform_discr(min_pt=-20, max_pt=20, shape=50)
    print("1D reconstruction")
    fwd_op = odl.IdentityOperator(domain)

    phantom = odl.phantom.cuboid(domain)
    data = fwd_op(phantom)
    data += odl.phantom.white_noise(fwd_op.range) * np.mean(data) * 0.1

    if reg == 'tgv':
        recons = tgv_denoise(domain, fwd_op, data, alpha=0.1, beta=1.0, niters=niters)
    elif reg == 'tv':
        recons = tv_denoise(domain, fwd_op, data, alpha=0.5, niters=niters)
    else:
        raise RuntimeError("Unknown regularizer: '%s'" % reg)

    # 1D plotting
    xs, = fwd_op.range.meshgrid
    true_img = fwd_op(phantom)
    true_data = true_img.data
    noisy_data = data.data
    recons_data = recons.data

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(xs, true_data, label='True')
    plt.plot(xs, noisy_data, label='Noisy')
    plt.plot(xs, recons_data, label='Recons')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return


def run_demo_2d(reg='tgv', niters=400):
    domain = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])
    print("2D b+w reconstruction")
    fwd_op = odl.IdentityOperator(domain)

    phantom = odl.phantom.tgv_phantom(domain)  # 2D demo
    data = fwd_op(phantom)
    data += odl.phantom.white_noise(fwd_op.range) * np.mean(data) * 0.1

    if reg == 'tgv':
        recons = tgv_denoise(domain, fwd_op, data, alpha=0.1, beta=1.0, niters=niters)
    elif reg == 'tv':
        recons = tv_denoise(domain, fwd_op, data, alpha=0.5, niters=niters)
    else:
        raise RuntimeError("Unknown regularizer: '%s'" % reg)

    # 2D plot demo
    xs, ys = fwd_op.range.meshgrid
    true_img = fwd_op(phantom)
    true_data = true_img.data[:, ::-1].T  # extract data, suitable for use in imshow
    noisy_data = data.data[:, ::-1].T
    recons_data = recons.data[:, ::-1].T
    cmap_to_use = 'gray'  # default = viridis
    extent = [xs[0, 0], xs[-1, 0], ys[0, 0], ys[0, -1]]

    norm = odl.NormOperator(domain)
    noisy_error = norm(true_img - noisy_data)
    recons_error = norm(true_img - recons)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 3, 1)  # extent = [left, right, bottom, top]
    plt.imshow(true_data, extent=extent, cmap=cmap_to_use)
    plt.title('True image')
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_data, extent=extent, cmap=cmap_to_use)
    plt.title('Noisy image (error = %g)' % noisy_error)
    plt.subplot(1, 3, 3)
    plt.imshow(recons_data, extent=extent, cmap=cmap_to_use)
    plt.title('Reconstruction (error = %g)' % recons_error)
    plt.show()
    return


def run_demo_2d_color(reg='tgv', niters=400):
    domain1 = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])
    domain = odl.ProductSpace(domain1, 3)  # 3 channels for colors
    print("2D color reconstruction")
    fwd_op = odl.IdentityOperator(domain)

    phantom = domain.element(3 * [odl.phantom.tgv_phantom(domain1)])
    data = fwd_op(phantom)
    data += odl.phantom.white_noise(fwd_op.range) * np.mean(data) * 0.5

    if reg == 'tgv':
        recons = tgv_denoise(domain, fwd_op, data, alpha=0.3, beta=1.0, niters=niters)
    elif reg == 'tv':
        recons = tv_denoise(domain, fwd_op, data, alpha=1.0, niters=niters)
    else:
        raise RuntimeError("Unknown regularizer: '%s'" % reg)
    # recons = domain.zero()  # temporary placeholder to avoid error-producing TV/TGV call

    # 2D color plot demo
    xs, ys = domain1.meshgrid
    true_img = fwd_op(phantom)
    norm = odl.NormOperator(domain)
    noisy_error = norm(true_img - data)
    recons_error = norm(true_img - recons)
    extent = [xs[0, 0], xs[-1, 0], ys[0, 0], ys[0, -1]]

    # Get images as numpy tensors in the correct layout (suitable for tgv phantom)
    # Note: for jpg images, use
    # true_img = np.clip(np.moveaxis(phantom.asarray(), [0, 1, 2], [2, 0, 1]), a_min=0.0, a_max=1.0)
    true_img = np.clip(np.moveaxis(phantom.asarray(), [0, 1, 2], [2, 1, 0]), a_min=0.0, a_max=1.0)[::-1,:,:]
    noisy_img = np.clip(np.moveaxis(data.asarray(), [0, 1, 2], [2, 1, 0]), a_min=0.0, a_max=1.0)[::-1,:,:]
    recons = np.clip(np.moveaxis(recons.asarray(), [0, 1, 2], [2, 1, 0]), a_min=0.0, a_max=1.0)[::-1,:,:]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    # plt.subplot(3, 3, 1)
    # plt.title('True image')
    # plt.imshow(true_img, extent=extent)
    # plt.subplot(3, 3, 2)
    # plt.imshow(noisy_img, extent=extent)
    # plt.title('Noisy image (error = %g)' % noisy_error)
    # plt.subplot(3, 3, 3)
    # plt.imshow(recons, extent=extent)
    # plt.title('Reconstruction (error = %g)' % recons_error)

    plt.subplot(3, 3, 1)
    plt.title('True image')
    tmp = true_img.copy()
    tmp[:, :, 1:] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 4)
    tmp = true_img.copy()
    tmp[:, :, [0, 2]] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 7)
    tmp = true_img.copy()
    tmp[:, :, :-1] = 0
    plt.imshow(tmp, extent=extent)

    plt.subplot(3, 3, 2)
    plt.title('Noisy image (error = %g)' % noisy_error)
    tmp = noisy_img.copy()
    tmp[:, :, 1:] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 5)
    tmp = noisy_img.copy()
    tmp[:, :, [0, 2]] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 8)
    tmp = noisy_img.copy()
    tmp[:, :, :-1] = 0
    plt.imshow(tmp, extent=extent)

    plt.subplot(3, 3, 3)
    plt.title('Reconstruction (error = %g)' % recons_error)
    tmp = recons.copy()
    tmp[:, :, 1:] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 6)
    tmp = recons.copy()
    tmp[:, :, [0, 2]] = 0
    plt.imshow(tmp, extent=extent)
    plt.subplot(3, 3, 9)
    tmp = recons.copy()
    tmp[:, :, :-1] = 0
    plt.imshow(tmp, extent=extent)

    plt.show()
    return


def main():
    reg = 'tgv'
    # reg = 'tv'
    niters = 400
    run_demo_1d(reg=reg, niters=niters)
    # run_demo_2d(reg=reg, niters=niters)
    # run_demo_2d_color(reg=reg, niters=niters)
    return


if __name__ == '__main__':
    main()
