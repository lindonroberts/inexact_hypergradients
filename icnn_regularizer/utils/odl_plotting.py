"""
Tools to plot ODL image objects
"""
import matplotlib.pyplot as plt
import numpy as np
import odl

__all__ = ['get_image_data', 'plot', 'sample_img']


def get_image_data(elem):
    """
    Get image data (domain and values) as numpy arrays

    In 1D:
        xs, data = get_image_data(elem)
    In 2D:
        xs, ys, data = get_image_data(elem)
    """
    domain = elem.space
    ndim = domain.ndim
    assert ndim in [1, 2], "get_image_data only supports 1D or 2D images"

    if ndim == 1:
        xs, = domain.meshgrid
        data = elem.data
        return xs, data
    else:  # ndim == 2
        xs, ys = domain.meshgrid
        data = elem.data[:, ::-1].T  # Rearrange data so that it can be used with plt.imshow()
        return xs, ys, data


def plot(elem, ax=None, ls='-', col=None, lbl=None, lw=1.0, cmap='gray'):
    """
    Make a plot of 'elem' (1D or 2D image) on the given matplotlib.pyplot axis (current axis if not specified).

    For 1D images, have extra plot information like label, marker size, etc.

    For 2D images, can specify colormap

    Return a dictionary of raw data, suitable for saving to disk
    """
    if ax is None:
        ax = plt.gca()

    ndim = elem.space.ndim
    assert ndim in [1, 2], "get_image_data only supports 1D or 2D images"

    if ndim == 1:
        xs, data = get_image_data(elem)
        ax.plot(xs, data, linestyle=ls, color=col, label=lbl, linewidth=lw)
        return {'xs': xs, 'data': data}
    else:  # ndim == 2
        xs, ys, data = get_image_data(elem)
        extent = [xs[0, 0], xs[-1, 0], ys[0, 0], ys[0, -1]]
        plt.imshow(data, extent=extent, cmap=cmap)
        return {'xs': xs[:,0], 'ys': ys[0,:], 'data': data}


def sample_img(ndim=1, npixels=100):
    # Build a very quick example image
    if ndim == 1:
        domain = odl.uniform_discr(min_pt=-20, max_pt=20, shape=npixels)
        phantom = odl.phantom.cuboid(domain)
    else:  # ndim == 2
        domain = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[npixels, npixels])
        phantom = odl.phantom.tgv_phantom(domain)
    fwd_op = odl.IdentityOperator(domain)
    img = fwd_op(phantom)
    img += odl.phantom.white_noise(fwd_op.range) * np.mean(img) * 0.1
    return domain, fwd_op, img


def main():
    plt.figure(1)
    plt.clf()

    # 1D image
    plt.subplot(1, 2, 1)
    domain, fwd_op, img = sample_img(ndim=1)
    img_data = plot(img)
    plt.grid()
    plt.title('1D example')
    print("1D data:")
    for key in img_data:
        print(' - %s = %s' % (key, str(img_data[key].shape)))
    print("")

    # 2D image
    plt.subplot(1, 2, 2)
    domain, fwd_op, img = sample_img(ndim=2)
    img_data = plot(img)
    plt.title('2D example')
    print("2D data:")
    for key in img_data:
        print(' - %s = %s' % (key, str(img_data[key].shape)))

    plt.show()
    return


if __name__ == '__main__':
    main()