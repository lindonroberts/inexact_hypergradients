"""
Implementation of a subsampling operator in ODL

Taken from
https://github.com/mehrhardt/Enhancing_hMRI/blob/main/code/misc.py
and originally used for
M. J. Ehrhardt, F. A. Gallagher, M. A. McLean, and C.-B. Schoenlieb, Enhancing the Spatial Resolution of Hyperpolarized
Carbon-13 MRI of Human  Brain Metabolism using Structure Guidance, 2021.
"""
import numpy as np
import odl
from skimage.measure import block_reduce as sk_block_reduce

__all__ = ['Subsampling']


class Subsampling(odl.Operator):
    '''  '''

    def __init__(self, domain, range, margin=None):
        """TBC

        Parameters
        ----------
        TBC

        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn((8, 8, 8))
        >>> Y = odl.rn((2, 2))
        >>> S = myOperators.Subsampling(X, Y)

        # Avg of 4x4 blocks in 1st two dimensions + all 8 components of last dimension
        """
        domain_shape = np.array(domain.shape)
        range_shape = np.array(range.shape)

        len_domain = len(domain_shape)
        len_range = len(range_shape)

        if margin is None:
            margin = 0

        if np.isscalar(margin):
            margin = [(margin, margin)] * len_domain

        self.margin = np.array(margin).astype('int')

        self.margin_index = []
        for m in self.margin:
            m0 = m[0]
            m1 = m[1]

            if m0 == 0:
                m0 = None

            if m1 == 0:
                m1 = None
            else:
                m1 = -m1

            self.margin_index.append((m0, m1))

        if len_domain < len_range:
            ValueError('TBC')
        else:
            if len_domain > len_range:
                range_shape = np.append(range_shape, np.ones(len_domain - len_range))

            self.block_size = tuple(((domain_shape - np.sum(self.margin, 1)) / range_shape).astype('int'))

        super(Subsampling, self).__init__(domain=domain, range=range,
                                          linear=True)

    def _call(self, x, out):
        m = self.margin_index
        if m is not None:
            if len(m) == 1:
                x = x[m[0][0]:m[0][1]]
            elif len(m) == 2:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1]]
            elif len(m) == 3:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]]
            else:
                ValueError('TBC')

        out[:] = np.squeeze(sk_block_reduce(x, block_size=self.block_size,
                                            func=np.mean))
        # block_reduce: returns Down-sampled image with same number of dimensions as input image.

    @property
    def adjoint(self):
        op = self

        class SubsamplingAdjoint(odl.Operator):

            def __init__(self, op):
                """TBC

                Parameters
                ----------
                TBC

                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))
                >>> S = myOperators.Subsampling(X, Y)

                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 15))
                >>> Y = odl.rn((2, 2))
                >>> S = myOperators.Subsampling(X, Y)

                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 15))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y)

                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))
                >>> S = myOperators.Subsampling(X, Y, margin=1)

                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 21))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y, margin=((0, 0),(0, 0),(3, 3)))
                """
                domain = op.range
                range = op.domain
                self.block_size = op.block_size
                self.margin = op.margin
                self.margin_index = op.margin_index

                x = range.zero()
                m = self.margin_index
                if m is not None:
                    if len(m) == 1:
                        x[m[0][0]:m[0][1]] = 1
                    elif len(m) == 2:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1]] = 1
                    elif len(m) == 3:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]] = 1
                    else:
                        ValueError('TBC')
                else:
                    x = range.one()

                self.factor = x.inner(range.one()) / domain.one().inner(domain.one())

                super(SubsamplingAdjoint, self).__init__(
                    domain=domain, range=range, linear=True)

            def _call(self, x, out):
                for i in range(len(x.shape), len(self.block_size)):
                    x = np.expand_dims(x, axis=i)

                if self.margin is None:
                    out[:] = np.kron(x, np.ones(self.block_size)) / self.factor
                else:
                    y = np.kron(x, np.ones(self.block_size)) / self.factor
                    out[:] = np.pad(y, self.margin, mode='constant')

            @property
            def adjoint(self):
                return op

        return SubsamplingAdjoint(self)

    @property
    def inverse(self):
        scaling = 1 / self.norm(estimate=True) ** 2
        return self.adjoint * scaling


def run_demo_1d():
    scale = 5
    domain = odl.uniform_discr(min_pt=-20, max_pt=20, shape=33 * scale)
    phantom = odl.phantom.cuboid(domain)

    # Need to make image dimensions a multiple of 4
    domain2 = odl.uniform_discr(min_pt=-20, max_pt=20, shape=32 * scale)
    phantom2 = domain2.element(phantom[:32 * scale])

    range2 = odl.uniform_discr(min_pt=-20, max_pt=20, shape=8 * scale)  # domain dimension / 4
    A = Subsampling(domain2, range2)
    out = A(phantom2)

    xs1, = domain2.meshgrid
    xs2, = range2.meshgrid
    p_data = phantom2.data
    out_data = out.data

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(xs1, p_data, label='In')
    plt.plot(xs2, out_data, label='Out')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return


def run_demo_2d():
    scale = 10
    domain = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[33 * scale, 43 * scale])
    # phantom = odl.phantom.cuboid(domain)
    phantom = odl.phantom.tgv_phantom(domain)

    # Need to make image dimensions a multiple of 4
    domain2 = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[32 * scale, 40 * scale])
    phantom2 = domain2.element(phantom[:32 * scale, :40 * scale])

    range2 = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[8 * scale, 10 * scale])  # domain dimension / 4
    A = Subsampling(domain2, range2)
    out = A(phantom2)

    # 2D plot demo
    xs1, ys1 = domain2.meshgrid
    xs2, ys2 = range2.meshgrid
    p_data = phantom2.data[:, ::-1].T
    out_data = out.data[:, ::-1].T  # extract data, suitable for use in imshow
    cmap_to_use = 'gray'  # default = viridis
    extent1 = [xs1[0, 0], xs1[-1, 0], ys1[0, 0], ys1[0, -1]]
    extent2 = [xs2[0, 0], xs2[-1, 0], ys2[0, 0], ys2[0, -1]]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)  # extent = [left, right, bottom, top]
    plt.imshow(p_data, extent=extent1, cmap=cmap_to_use)
    plt.title('Input')
    plt.subplot(1, 2, 2)
    plt.imshow(out_data, extent=extent2, cmap=cmap_to_use)
    plt.title('Output')
    plt.show()
    return


def main():
    run_demo_1d()
    run_demo_2d()
    return


if __name__ == '__main__':
    main()
