"""
Construct ODL train/test datasets for different problem classes
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import odl
import os
import torch

try:
    from .odl_plotting import plot
    from .odl_subsampling import Subsampling
    from .odl_torch_wrapper import OperatorModule
except ModuleNotFoundError:
    from odl_plotting import plot
    from odl_subsampling import Subsampling
    from odl_torch_wrapper import OperatorModule

__all__ = ['DenoisingDataset1D', 'DenoisingDataset2D', 'SuperresolutionDataset2D', 'numpy_to_tensor']

# Indices for training/test pairs, as returned by generate_single_image_pair()
ODL_TRUE_IMG = 0
ODL_NOISY_DATA = 1
TRUE_IMG = 2
NOISY_DATA = 3


def numpy_to_tensor(img, complex_data=False):
    # Take a numpy array and make a torch tensor
    # But torch modules expect one extra dimension (for minibatching), so do that here too
    X = np.zeros((1,) + img.shape, dtype=complex if complex_data else float)
    X[0] = np.array(img)
    return torch.from_numpy(X)  #.double()


class AbstractDataset(ABC):
    """
    Abstract base class for all dataset objects
    """
    def __init__(self, xmin, xmax, npixels, seed, ntrain, ntest, noise_level, use_true_img_as_initial_recons):
        super(AbstractDataset, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.npixels = npixels
        self.seed = seed
        self.ntrain = ntrain
        self.ntest = ntest
        self.noise_level = noise_level
        self.use_true_img_as_initial_recons = use_true_img_as_initial_recons
        self.device = 'cpu'

        # Placeholders
        self.training_data = []
        self.test_data = []
        self.training_recons = []
        self.test_recons = []
        self.domain_norm = None
        self.range_norm = None
        self.fwd_op = None
        self.fwd_op_norm = 1.0
        return

    @abstractmethod
    def build_spaces(self):  # generate domain/range/fwd_op, etc
        pass

    @abstractmethod
    def generate_data(self):  # generate a set of training/test data
        pass

    def reset_reconstructions(self):
        # Make available reconstructions some generic starting value
        if self.use_true_img_as_initial_recons:
            # Don't quite return exact values, as this causes differentiability problems for some parts of the objective
            self.training_recons = [1.001*torch.clone(self.training_data[i][TRUE_IMG]) for i in range(self.ntrain)]
            self.test_recons = [1.001*torch.clone(self.test_data[i][TRUE_IMG]) for i in range(self.ntest)]
        else:
            self.training_recons = [torch.ones_like(self.training_data[i][TRUE_IMG]) for i in range(self.ntrain)]
            self.test_recons = [torch.ones_like(self.test_data[i][TRUE_IMG]) for i in range(self.ntest)]
        return

    def get_noisy_img(self, i, training_data=True, as_tensor=True):
        nimg = self.ntrain if training_data else self.ntest
        assert 0 <= i < nimg, "get_recons: bad i (got %g, expect in [0,...,%g]" % (i, nimg - 1)
        return self.training_data[i][NOISY_DATA if as_tensor else ODL_NOISY_DATA] if training_data else self.test_data[i][NOISY_DATA if as_tensor else ODL_NOISY_DATA]

    def get_recons(self, i, training_data=True):
        nimg = self.ntrain if training_data else self.ntest
        assert 0 <= i < nimg, "get_recons: bad i (got %g, expect in [0,...,%g]" % (i, nimg - 1)
        return self.training_recons[i] if training_data else self.test_recons[i]

    def upper_objfun(self, i, img_recons=None, training_data=True):
        # i-th upper-level objective
        nimg = self.ntrain if training_data else self.ntest
        assert 0 <= i < nimg, "get_recons: bad i (got %g, expect in [0,...,%g]" % (i, nimg - 1)
        true_img = self.training_data[i][TRUE_IMG] if training_data else self.test_data[i][TRUE_IMG]
        if img_recons is None:
            img_recons = self.training_recons[i] if training_data else self.test_recons[i]
        return self.domain_norm(img_recons - true_img)**2

    def upper_gradient_wrt_img_recons(self, i, training_data=True):
        # Gradient of i-th upper-level objective w.r.t img_recons
        img_recons_var = torch.autograd.Variable(self.training_recons[i], requires_grad=True)  # flag that you want derivatives wrt these values
        loss = self.upper_objfun(i, img_recons_var, training_data=training_data)
        return torch.autograd.grad(loss, img_recons_var)[0]  # same shape as img_recons

    def lower_level_data_fit_l2_loss(self, x, i, training_data=True):
        # Data fitting term used for lower-level problem
        noisy_img = self.get_noisy_img(i, training_data=training_data, as_tensor=True)
        return self.range_norm(self.fwd_op(x) - noisy_img) ** 2

    @abstractmethod
    def plot(self):  # make a plot of the training or test data and/or reconstructions
        pass


class DenoisingDataset1D(AbstractDataset):
    """
    Collection of training and test data for the 1D denoising problem
    """
    def __init__(self, xmin=0.0, xmax=1.0, npixels=64, seed=0, ntrain=6, ntest=6, noise_level=0.2, img_type='rect', use_true_img_as_initial_recons=False):
        super(DenoisingDataset1D, self).__init__(xmin, xmax, npixels, seed, ntrain, ntest, noise_level, use_true_img_as_initial_recons)
        # Input information
        self.pwlinear_nsegments = 4

        # Type of image to use
        self.img_type = img_type
        assert self.img_type in ['rect', 'pw_linear'], "Unknown img_type '%s'" % self.img_type

        # Build domain/range spaces and relevant operators
        self.build_spaces()

        # Create train/test datasets and some default reconstructions
        self.generate_data()
        self.reset_reconstructions()
        return

    def build_spaces(self):
        # Domain
        self.domain = odl.uniform_discr(min_pt=self.xmin, max_pt=self.xmax, shape=self.npixels)

        # Forward operator
        self.odl_fwd_op = odl.IdentityOperator(self.domain)  # in ODL
        self.fwd_op = OperatorModule(self.odl_fwd_op).to(self.device)       # in torch
        self.fwd_op_norm = 1.0  # used to calculate Lipschitz constant
        self.range = self.odl_fwd_op.range

        # Domain and range norms, for upper- and lower-level objectives respectively
        self.odl_domain_norm = odl.NormOperator(self.domain)
        self.domain_norm = OperatorModule(self.odl_domain_norm).to(self.device)
        self.odl_range_norm = odl.NormOperator(self.range)
        self.range_norm = OperatorModule(self.odl_range_norm).to(self.device)
        return

    def generate_single_image_pair(self, as_tensor=True):
        if self.img_type == 'rect':
            # Rectangular phantom
            mean = 0.5 * (self.xmin + self.xmax)
            r1 = 0.5 * (self.xmin + mean)
            r2 = 0.5 * (mean + self.xmax)
            center = np.random.uniform(r1, r2)
            radius = np.random.uniform(0.5 * r1, r1)
            img = odl.phantom.cuboid(self.domain, (center - radius,), (center + radius,))

        elif self.img_type == 'pw_linear':
            # Piecewise linear image with jumps
            base_img = self.domain.zero()
            img_data = np.zeros(base_img.asarray().shape)
            xs, = self.domain.meshgrid
            nsegments = self.pwlinear_nsegments
            segment_len = self.npixels // nsegments
            slopes = np.random.uniform(low=-10.0, high=10.0, size=(nsegments,))
            jumps = np.random.uniform(low=-1.0, high=1.0, size=(nsegments - 1,))
            for i in range(nsegments):
                intercept = 0.0 if i == 0 else img_data[i * segment_len - 1] + jumps[i - 1]
                img_data[(i * segment_len):] = intercept + slopes[i] * xs[:(nsegments - i) * segment_len]

            img = self.domain.element(img_data)

        else:
            raise RuntimeError("Unknown img_type: '%s'" % self.img_type)

        # Training data is fwd_op(img) + noise
        data = self.odl_fwd_op(img)
        data += odl.phantom.white_noise(self.range) * max(np.mean(np.abs(data)), 1.0) * self.noise_level

        if as_tensor:
            # Convert to torch tensor of suitable dimension (otherwise keep as ODL objects)
            img_tensor = numpy_to_tensor(img.data, complex_data=False).reshape((1, 1, self.npixels))
            data_tensor = numpy_to_tensor(data.data, complex_data=False).reshape((1, 1, self.npixels))
            return img, data, img_tensor, data_tensor
        else:
            return img, data

    def generate_data(self):
        np.random.seed(self.seed)
        # training_data and test_data are lists of (img, data) pairs
        self.training_data = [self.generate_single_image_pair(as_tensor=True) for _ in range(self.ntrain)]
        self.test_data = [self.generate_single_image_pair(as_tensor=True) for _ in range(self.ntest)]
        return

    def plot(self, title=None, nrows=2, training_data=True):
        # Decide how many columns are needed to plot everything
        nimgs = self.ntrain if training_data else self.ntest
        ncols = nimgs // nrows
        if ncols * nrows < nimgs:
            ncols += 1

        plt.figure(figsize=(3*ncols, 3*nrows))
        plt.clf()

        total_obj = 0.0
        for i in range(nimgs):
            this_img = self.training_data[i] if training_data else self.test_data[i]
            true_img = this_img[ODL_TRUE_IMG]
            noisy_data = this_img[ODL_NOISY_DATA]
            recons_tensor = self.training_recons[i] if training_data else self.test_recons[i]
            recons = self.domain.element(recons_tensor.flatten())
            obj = float(self.upper_objfun(i, training_data=training_data))
            total_obj += obj

            plt.subplot(nrows, ncols, i + 1)
            plot(true_img, ls='-', col='C0', lw=2.0, lbl='True data')
            plot(noisy_data, ls='solid', col='C1', lw=1.5, lbl='Noisy data')
            plot(recons, ls='solid', col='C3', lw=2.0, lbl='Reconstruction')
            plt.grid()
            if i == 0:
                plt.legend(loc='best')
            if obj is not None:
                plt.xlabel('Loss = %g' % obj)

        # Put title on first img
        plt.subplot(nrows, ncols, 1)
        if title is not None:
            plt.title(title)
        else:
            plt.title('Mean loss = %g' % (total_obj/nimgs))
        plt.tight_layout()
        return


def get_bsds_ids(infolder='BSDS300'):
    # Load list of train and test ids
    train_ids = np.loadtxt(os.path.join(infolder, 'iids_train.txt'), dtype=int)
    test_ids = np.loadtxt(os.path.join(infolder, 'iids_test.txt'), dtype=int)
    return train_ids, test_ids


def crop_img(img, crop_dim=None, max_h=None, max_w=None):
    # Select the middle part of a 2D image (B&W or color both work)
    if crop_dim is None:
        return img
    else:
        assert len(crop_dim) == 2, "crop_dim must be 2-tuple (height, width)"
        crop_h, crop_w = crop_dim
        crop_h, crop_w = int(crop_h), int(crop_w)
        if max_h is not None:
            assert 1 <= crop_h <= max_h, "crop_h must be in [1,%g]" % max_h
        if max_w is not None:
            assert 1 <= crop_w <= max_w, "crop_w must be in [1,%g]" % max_w
        h, w = img.shape[0], img.shape[1]
        x1 = (h - crop_h) // 2
        x2 = x1 + crop_h
        y1 = (w - crop_w) // 2
        y2 = y1 + crop_w
        if len(img.shape) == 2:  # 2D B&W
            return img[x1:x2, y1:y2]
        else:  # 2D color
            return img[x1:x2, y1:y2, :]


def read_bsds_images(ntrain, ntest, infolder='BSDS300', crop_dim=None):
    # All images have height and width in [321,..., 481]
    # Pixelintensities are in [0,255], so scale to [0,1]
    max_w, max_h = 321, 321
    train_ids, test_ids = get_bsds_ids(infolder=infolder)
    train_data = []
    for i in range(ntrain):
        img_file = os.path.join(infolder, 'images', 'train', '%g.jpg' % train_ids[i])
        train_data.append(crop_img(plt.imread(img_file), crop_dim=crop_dim, max_w=max_w, max_h=max_h)/255.0)
    test_data = []
    for i in range(ntest):
        img_file = os.path.join(infolder, 'images', 'test', '%g.jpg' % test_ids[i])
        test_data.append(crop_img(plt.imread(img_file), crop_dim=crop_dim, max_w=max_w, max_h=max_h)/255.0)
    return train_ids[:ntrain], test_ids[:ntest], train_data, test_data


class DenoisingDataset2D(AbstractDataset):
    """
    Collection of training and test data for the 2D denoising problem
    """
    def __init__(self, npixels=64, seed=0, ntrain=6, ntest=6, noise_level=0.2, infolder='BSDS300', use_true_img_as_initial_recons=False):
        super(DenoisingDataset2D, self).__init__(0.0, 1.0, npixels, seed, ntrain, ntest, noise_level, use_true_img_as_initial_recons)
        # Build domain/range spaces and relevant operators
        self.build_spaces()

        # Create train/test datasets and some default reconstructions
        self.generate_data(infolder=infolder)
        self.reset_reconstructions()
        return

    def build_spaces(self):
        # Domain
        self.domain1 = odl.uniform_discr(min_pt=[self.xmin, self.xmin], max_pt=[self.xmax, self.xmax], shape=[self.npixels, self.npixels])
        self.domain = odl.ProductSpace(self.domain1, 3)  # 3 channels for colors

        # Forward operator
        self.odl_fwd_op = odl.IdentityOperator(self.domain)  # in ODL
        self.fwd_op = OperatorModule(self.odl_fwd_op).to(self.device)  # in torch
        self.fwd_op_norm = 1.0  # used to calculate Lipschitz constant
        self.range = self.odl_fwd_op.range

        # Domain and range norms, for upper- and lower-level objectives respectively
        self.odl_domain_norm = odl.NormOperator(self.domain)
        self.domain_norm = OperatorModule(self.odl_domain_norm).to(self.device)
        self.odl_range_norm = odl.NormOperator(self.range)
        self.range_norm = OperatorModule(self.odl_range_norm).to(self.device)
        return

    def add_noise(self, img):
        # Training data is fwd_op(img) + noise
        data = self.odl_fwd_op(img)
        return data + odl.phantom.white_noise(self.range) * max(np.mean(np.abs(data)), 1.0) * self.noise_level

    def odl_element_from_numpy(self, img):
        return self.domain.element([img[:, :, i] for i in range(3)])

    def odl_element_from_tensor(self, img):
        return self.domain.element([img[0, i, :, :] for i in range(3)])

    def generate_data(self, infolder='BSDS300'):
        np.random.seed(self.seed)
        # training_data and test_data are lists of (img, data) pairs
        train_idx, test_idx, train_data, test_data = read_bsds_images(ntrain=self.ntrain, ntest=self.ntest,
                                                                      crop_dim=(self.npixels, self.npixels),
                                                                      infolder=infolder)
        # Store image filenames for reference
        self.train_idx = train_idx
        self.test_idx = test_idx
        # Convert train_data and test_data to ODL product space elements
        odl_train_data = [self.odl_element_from_numpy(train_data[i]) for i in range(self.ntrain)]
        odl_test_data = [self.odl_element_from_numpy(test_data[i]) for i in range(self.ntest)]
        # Add noise
        odl_train_noisy = [self.add_noise(img) for img in odl_train_data]
        odl_test_noisy = [self.add_noise(img) for img in odl_test_data]
        # Convert all data to torch tensors
        self.training_data = [(odl_train_data[i], odl_train_noisy[i], numpy_to_tensor(odl_train_data[i], complex_data=False), numpy_to_tensor(odl_train_noisy[i], complex_data=False)) for i in range(self.ntrain)]
        self.test_data = [(odl_test_data[i], odl_test_noisy[i], numpy_to_tensor(odl_test_data[i], complex_data=False), numpy_to_tensor(odl_test_noisy[i], complex_data=False)) for i in range(self.ntest)]
        return

    def plot(self, title=None, nrows=2, training_data=True, img_type='orig'):
        assert img_type in ['orig', 'noisy', 'recons'], "Unknown img_type: '%s'" % img_type

        # Decide how many columns are needed to plot everything
        nimgs = self.ntrain if training_data else self.ntest
        ncols = nimgs // nrows
        if ncols * nrows < nimgs:
            ncols += 1

        plt.figure(figsize=(3*ncols, 3*nrows))
        plt.clf()

        total_obj = 0.0
        for i in range(nimgs):
            obj = None
            if img_type == 'orig':
                this_img = self.training_data[i][ODL_TRUE_IMG] if training_data else self.test_data[i][ODL_TRUE_IMG]
            elif img_type == 'noisy':
                this_img = self.training_data[i][ODL_NOISY_DATA] if training_data else self.test_data[i][ODL_NOISY_DATA]
                obj = float(self.upper_objfun(i, img_recons=numpy_to_tensor(this_img), training_data=training_data))
            elif img_type == 'recons':
                this_img = self.training_recons[i] if training_data else self.test_recons[i]
                this_img = self.odl_element_from_tensor(this_img)
                # Need to convert from tensor to ODL image
                obj = float(self.upper_objfun(i, training_data=training_data))
            else:
                raise RuntimeError("Unknown img_type: '%s'" % img_type)

            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(np.clip(np.moveaxis(this_img.asarray(), [0, 1, 2], [2, 0, 1]), a_min=0.0, a_max=1.0))
            plt.xticks([], [])
            plt.yticks([], [])
            if img_type == 'orig':
                plt.xlabel('%g.jpg' % self.train_idx[i] if training_data else self.test_idx[i])
            else:
                total_obj += obj
                plt.xlabel('Loss = %g' % obj)

        # Put title on first img
        plt.subplot(nrows, ncols, 1)
        if title is not None:
            plt.title(title)
        elif img_type == 'orig':
            plt.title('Original images')
        else:
            plt.title('Mean loss = %g' % (total_obj/nimgs))
        plt.tight_layout()
        return


class SuperresolutionDataset2D(AbstractDataset):
    """
    Collection of training and test data for the 2D superresolution problem
    """
    def __init__(self, npixels=64, seed=0, ntrain=6, ntest=6, noise_level=0.0, subsample_factor=4, infolder='BSDS300', use_true_img_as_initial_recons=False):
        super(SuperresolutionDataset2D, self).__init__(0.0, 1.0, npixels, seed, ntrain, ntest, noise_level, use_true_img_as_initial_recons)
        # Input information
        self.subsample_factor = subsample_factor

        assert self.npixels % self.subsample_factor == 0, "npixels (%g) must be a multiple of subsample_factor (%g)" % (self.npixels, self.subsample_factor)

        # Build domain/range spaces and relevant operators
        self.build_spaces()

        # Create train/test datasets and some default reconstructions
        self.generate_data(infolder)
        self.reset_reconstructions()
        return

    def build_spaces(self):
        # Domain
        self.domain1 = odl.uniform_discr(min_pt=[self.xmin, self.xmin], max_pt=[self.xmax, self.xmax], shape=[self.npixels, self.npixels])
        self.domain = odl.ProductSpace(self.domain1, 3)  # 3 channels for colors

        # Range (subsampled space)
        self.range1 = odl.uniform_discr(min_pt=[self.xmin, self.xmin], max_pt=[self.xmax, self.xmax],
                                         shape=[self.npixels // self.subsample_factor,
                                                self.npixels // self.subsample_factor])
        self.range = odl.ProductSpace(self.range1, 3)  # 3 channels for colors

        # Forward operator
        self.odl_fwd_op = Subsampling(self.domain, self.range)  # in ODL
        self.fwd_op = OperatorModule(self.odl_fwd_op).to(self.device)  # in torch
        self.fwd_op_norm = 1.1 * odl.power_method_opnorm(self.odl_fwd_op)  # used to calculate Lipschitz constant

        # Domain and range norms, for upper- and lower-level objectives respectively
        self.odl_domain_norm = odl.NormOperator(self.domain)
        self.domain_norm = OperatorModule(self.odl_domain_norm).to(self.device)
        self.odl_range_norm = odl.NormOperator(self.range)
        self.range_norm = OperatorModule(self.odl_range_norm).to(self.device)
        return

    def downsample(self, img):
        # Training data is fwd_op(img) + noise [default noise level is zero]
        data = self.odl_fwd_op(img)
        return data + odl.phantom.white_noise(self.range) * max(np.mean(np.abs(data)), 1.0) * self.noise_level

    def odl_element_from_numpy(self, img):
        return self.domain.element([img[:, :, i] for i in range(3)])

    def odl_element_from_tensor(self, img):
        return self.domain.element([img[0, i, :, :] for i in range(3)])

    def generate_data(self, infolder):
        np.random.seed(self.seed)
        # training_data and test_data are lists of (img, data) pairs
        train_idx, test_idx, train_data, test_data = read_bsds_images(ntrain=self.ntrain, ntest=self.ntest,
                                                                      crop_dim=(self.npixels, self.npixels),
                                                                      infolder=infolder)
        # Store image filenames for reference
        self.train_idx = train_idx
        self.test_idx = test_idx
        # Convert train_data and test_data to ODL product space elements
        odl_train_data = [self.odl_element_from_numpy(train_data[i]) for i in range(self.ntrain)]
        odl_test_data = [self.odl_element_from_numpy(test_data[i]) for i in range(self.ntest)]
        # Add noise
        odl_train_noisy = [self.downsample(img) for img in odl_train_data]
        odl_test_noisy = [self.downsample(img) for img in odl_test_data]
        # Convert all data to torch tensors
        self.training_data = [(odl_train_data[i], odl_train_noisy[i], numpy_to_tensor(odl_train_data[i], complex_data=False), numpy_to_tensor(odl_train_noisy[i], complex_data=False)) for i in range(self.ntrain)]
        self.test_data = [(odl_test_data[i], odl_test_noisy[i], numpy_to_tensor(odl_test_data[i], complex_data=False), numpy_to_tensor(odl_test_noisy[i], complex_data=False)) for i in range(self.ntest)]
        return

    def plot(self, title=None, nrows=2, training_data=True, img_type='orig'):
        assert img_type in ['orig', 'noisy', 'recons'], "Unknown img_type: '%s'" % img_type

        # Decide how many columns are needed to plot everything
        nimgs = self.ntrain if training_data else self.ntest
        ncols = nimgs // nrows
        if ncols * nrows < nimgs:
            ncols += 1

        plt.figure(figsize=(3*ncols, 3*nrows))
        plt.clf()

        total_obj = 0.0
        for i in range(nimgs):
            obj = None
            if img_type == 'orig':
                this_img = self.training_data[i][ODL_TRUE_IMG] if training_data else self.test_data[i][ODL_TRUE_IMG]
            elif img_type == 'noisy':
                this_img = self.training_data[i][ODL_NOISY_DATA] if training_data else self.test_data[i][ODL_NOISY_DATA]
                # can't calculate obj here as domain != range
            elif img_type == 'recons':
                this_img = self.training_recons[i] if training_data else self.test_recons[i]
                this_img = self.odl_element_from_tensor(this_img)
                # Need to convert from tensor to ODL image
                obj = float(self.upper_objfun(i, training_data=training_data))
            else:
                raise RuntimeError("Unknown img_type: '%s'" % img_type)

            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(np.clip(np.moveaxis(this_img.asarray(), [0, 1, 2], [2, 0, 1]), a_min=0.0, a_max=1.0))
            plt.xticks([], [])
            plt.yticks([], [])
            if img_type == 'orig':
                plt.xlabel('%g.jpg' % self.train_idx[i] if training_data else self.test_idx[i])
            elif img_type == 'noisy':
                plt.xlabel('%g.jpg (subsampled x%g)' % (self.train_idx[i] if training_data else self.test_idx[i], self.subsample_factor))
            else:
                total_obj += obj
                plt.xlabel('Loss = %g' % obj)

        # Put title on first img
        plt.subplot(nrows, ncols, 1)
        if title is not None:
            plt.title(title)
        elif img_type == 'orig':
            plt.title('Original images')
        elif img_type == 'noisy':
            plt.title('Subsampled images')
        else:
            plt.title('Mean loss = %g' % (total_obj/nimgs))
        plt.tight_layout()
        return


def main():
    # dataset = DenoisingDataset2D(npixels=200)
    dataset = SuperresolutionDataset2D(npixels=200)
    print(dataset.domain)
    print(dataset.domain.ndim)
    return
    # dataset.plot(training_data=True, img_type='orig')
    dataset.plot(training_data=True, img_type='noisy')
    # dataset.plot(training_data=True, img_type='recons')
    plt.show()
    return
    # dataset = DenoisingDataset1D(img_type='rect')
    dataset = DenoisingDataset1D(img_type='pw_linear')
    dataset.plot(training_data=True)
    plt.show()
    # plt.savefig('train.png', bbox_inches='tight')
    dataset.plot(training_data=False)
    plt.show()
    # plt.savefig('test.png', bbox_inches='tight')
    return


if __name__ == '__main__':
    main()
