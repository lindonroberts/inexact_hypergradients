"""
Convex regularizer functions as torch models
"""

from abc import ABC, abstractmethod
import numpy as np
import torch

try:
    from odl_dataset import AbstractDataset
except ModuleNotFoundError:
    from .odl_dataset import AbstractDataset

__all__ = ['BaseConvexRegularizer']


def return_vector(v, as_numpy=False):
    return v.detach().numpy() if as_numpy else v


class BaseConvexRegularizer(torch.nn.Module, ABC):
    """
    Abstract class for convex regularizer models
    """
    def __init__(self, dataset: AbstractDataset):
        super(BaseConvexRegularizer, self).__init__()
        self.dataset = dataset
        self.current_index = 0  # which image in the dataset are we currently reconstructing?
        self.current_recons_training_data = True  # are we reconstructing the training or test image?

    def data_sq_loss(self, x, i=None, training_data=None):
        if i is None:
            i = self.current_index
        if training_data is None:
            training_data = self.current_recons_training_data
        return self.dataset.lower_level_data_fit_l2_loss(x, i, training_data=training_data)

    @abstractmethod
    def lip_const(self):  # Lipschitz constant of gradient
        pass

    @abstractmethod
    def convex_const(self):  # strong convexity constant
        pass

    @abstractmethod
    def forward(self, X):  # forward operator
        pass

    @abstractmethod
    def project_weights_to_feasible_set(self):  # ensure weights feasible
        pass

    def __repr__(self):
        return super(BaseConvexRegularizer, self).__repr__()

    def __str__(self):
        return super(BaseConvexRegularizer, self).__str__()

    def __call__(self, X):
        return super(BaseConvexRegularizer, self).__call__(X)

    def parameters(self, recurse=True):
        return super(BaseConvexRegularizer, self).parameters(recurse=recurse)

    def zero_grad(self, set_to_none=False):
        return super(BaseConvexRegularizer, self).zero_grad(set_to_none=set_to_none)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super(BaseConvexRegularizer, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return super(BaseConvexRegularizer, self).load_state_dict(state_dict, strict=strict)

    def numel(self):
        return sum(param.numel() for param in super(BaseConvexRegularizer, self).parameters() if param.requires_grad)

    def objfun(self, X, as_numpy=False):
        # Evaluate forward model, but allow return as NumPy array
        return return_vector(self.forward(X), as_numpy=as_numpy)

    def parameter_values(self, as_numpy=False):
        param_list = [param.flatten() for param in super(BaseConvexRegularizer, self).parameters() if param.requires_grad]
        return return_vector(torch.cat(param_list, dim=0), as_numpy=as_numpy)

    def parameter_gradients(self, as_numpy=False):
        try:
            grad_list = [param.grad.flatten() for param in super(BaseConvexRegularizer, self).parameters() if param.requires_grad]
        except AttributeError:
            return None  # gradients not initialized
        return return_vector(torch.cat(grad_list, dim=0), as_numpy=as_numpy)

    def gradient_wrt_parameters(self, X, as_numpy=False):
        self.zero_grad()
        Rval = self.forward(X)
        Rval.backward()
        return self.parameter_gradients(as_numpy=as_numpy)

    def gradient_wrt_inputs(self, X, as_numpy=False):
        self.zero_grad()
        Xvar = torch.autograd.Variable(X, requires_grad=True)
        Rval = self.forward(Xvar)
        return return_vector(torch.autograd.grad(Rval, Xvar)[0], as_numpy=as_numpy)

    def hess_vec_wrt_inputs(self, Xs, v, as_numpy=False):
        # Given a torch model, y=model(x,w) with fixed weights w, evaluate hess_x model(x)*v, evaluated at each input in batch Xs
        # Trick: hessf(x)*v = grad_x [ gradf(x)^T v ]
        # https://discuss.pytorch.org/t/calculating-hessian-vector-product/11240/3
        # Output is a tensor, same shape as Xs
        if len(Xs.shape) == len(v.shape):
            assert Xs.shape == v.shape, "Mismatch between Ys shape (%s) and v shape (%s)" % (str(Xs.shape), str(v.shape))
            Ys_var = torch.autograd.Variable(Xs, requires_grad=True)  # flag that you want derivatives wrt these values
            self.zero_grad()
            hess_vec = torch.autograd.functional.vhp(func=self.forward, inputs=Ys_var, v=v)[1]
        else:
            # first index of Ys is over each batch
            assert Xs[0, ...].shape == v.shape, "Mismatch between Ys shape (%s) and v shape (%s)" % (str(Xs.shape), str(v.shape))
            nbatch = Xs.shape[0]
            Ys_var = torch.autograd.Variable(Xs, requires_grad=True)  # flag that you want derivatives wrt these values
            hess_vec = torch.zeros_like(Xs)
            for i in range(nbatch):
                self.zero_grad()
                Hv = torch.autograd.functional.vhp(func=self.forward, inputs=Ys_var[i, ...], v=v)[1]
                hess_vec[i, ...] = Hv
        return return_vector(hess_vec, as_numpy=as_numpy)

    def hessian(self, Xs, as_numpy=False):
        # Full Hessian d_X^2 objfun(X,w), size |x|*|x|
        nX = Xs.numel()
        H_rows = []
        for i in range(nX):
            v = np.zeros((nX,))
            v[i] = 1.0
            v = torch.reshape(torch.from_numpy(v), Xs.shape)
            Hv = self.hess_vec_wrt_inputs(Xs, v, as_numpy=as_numpy)
            H_rows.append(Hv)
        if as_numpy:
            H = np.array(H_rows)
        else:
            H = torch.transpose(torch.stack(H_rows, dim=-1), 0, 1)
        return H

    def jac_vec(self, Xs, v, batch_idx=0, as_numpy=False):
        """
        Given a torch model, y=model(x,w) with fixed weights w, define the Jacobian as
            J(x,w) = d_w d_x model(x, w), of size |x|*|w|
        Here, evaluate v.T * J(x,w) for a single batch element x = Xs[batch_idx,...].
        Note: v needs same dimensions as each batch element Xs[i,...]

        This computation is based on the trick:
            v.T * J(x,w) = grad_w [ grad_x(model)^T v ]

        By default, there is no output to this function (as per usual torch training optimization interface),
        but setting output_vector=True produces the vector as expected.
        """
        if len(Xs.shape) == len(v.shape):
            Xs_var = torch.autograd.Variable(Xs, requires_grad=True)  # flag that you want derivatives wrt these values
            self.zero_grad()
            output = self.forward(Xs_var)
            grad_model = torch.autograd.grad(output, Xs_var, create_graph=True)[0]
            dir_deriv = (grad_model * v).sum()  # computing dot product explicitly to avoid dimension mismatch issues for tensors
            dir_deriv.backward()
        else:  # with minibatching
            nbatch = Xs.shape[0]
            assert batch_idx < nbatch, "batch_idx (%g) must be smaller than batch size (%g)" % (batch_idx, nbatch)
            assert Xs[batch_idx, ...].shape == v.shape, "Mismatch between Xs shape (%s) and v shape (%s)" % (str(Xs[batch_idx, ...].shape), str(v.shape))
            Xs_var = torch.autograd.Variable(Xs, requires_grad=True)  # flag that you want derivatives wrt these values
            self.zero_grad()
            output = self.forward(Xs_var)
            grad_model = torch.autograd.grad(output[batch_idx], Xs_var, create_graph=True)[0]
            dir_deriv = (grad_model[batch_idx, ...] * v).sum()  # computing dot product explicitly to avoid dimension mismatch issues for tensors
            dir_deriv.backward()
        return self.parameter_gradients(as_numpy=as_numpy)

    def jacobian(self, Xs, batch_idx=0, as_numpy=False):
        # Full Jacobian d_X d_w objfun(X,w), size |x|*|w|
        nX = Xs.numel()
        J_rows = []
        for i in range(nX):
            v = np.zeros((nX,))
            v[i] = 1.0
            v = torch.reshape(torch.from_numpy(v), Xs.shape)
            vT_J = self.jac_vec(Xs, v, batch_idx=batch_idx, as_numpy=as_numpy)
            J_rows.append(vT_J)
        if as_numpy:
            J = np.array(J_rows)
        else:
            J = torch.transpose(torch.stack(J_rows, dim=-1), 0, 1)
        return J

    # Decorator avoids RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    # https://discuss.pytorch.org/t/write-to-data-instead-runtimeerror-a-leaf-variable-that-requires-grad-has-been-used-in-an-in-place-operation/41049
    # Same as torch L-BFGS implementation of step(self, ...)
    @torch.no_grad()
    def update_model_params(self, step_dir, step_size=1.0):
        '''
        Update model parameters by a given step direction (which should be a tensor with length = number of parameters)

        model.params = model.params + step_size * step_dir

        Based on L-BFGS updating (https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py)
        '''
        assert len(step_dir) == self.numel(), "Mismatch: step_dir has %g elements, model expects %g" % (len(step_dir), self.numel())
        offset = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue  # skip this one
            numel = p.numel()
            p.add_(step_dir[offset:offset + numel].view_as(p), alpha=step_size)  # view as to avoid deprecated pointwise semantics
            offset += numel
        return

    @torch.no_grad()
    def set_params(self, new_param_values):
        current_params = self.parameter_values(as_numpy=False)
        self.update_model_params(new_param_values - current_params)
        return


class L2Regularizer(BaseConvexRegularizer):
    """
    Class implementing L2 penalty term

    Has a single parameter w (self.l2_penalty). Model is
        L2net(x) = 0.5*c(w)*||x||_2^2
    where
        c(w) = softplus(w) = log(1+exp(w))
    maps any real w to c(w)>0 via smoothed ReLU approximation.
    See https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html

    Note: c(w) ~ 0 as w -> -inf, and c(w) ~ w as w -> +inf
    Note: c'(w) = exp(w) / (exp(w)+1) = 1 / (1+exp(-w))
    """
    def __init__(self, start_value=-5.0):
        super(L2Regularizer, self).__init__()
        self.penalty_param = torch.nn.Parameter((start_value) * torch.ones(1))

    def l2_weighting(self, as_float=False):  # c(self.l2_penalty)
        c = torch.nn.functional.softplus(self.penalty_param)
        return float(c) if as_float else c

    def deriv_l2_weighting(self, as_float=False):  # gradient of self.l2_weighting() with respect to params
        d = float(torch.special.expit(self.penalty_param))
        return d if as_float else torch.from_numpy(np.array([d]))

    def lip_const(self):
        # Lipschitz constant of gradient of model (wrt inputs x)
        return self.l2_weighting(as_float=True)

    def convex_const(self):
        # Strong convexity constant of model (wrt inputs x)
        return self.l2_weighting(as_float=True)

    def project_weights_to_feasible_set(self):
        # Nothing needed
        return self.parameters()

    def forward(self, x):
        l2_sqnorm = torch.sum(x.view(x.size(0), -1) ** 2, dim=1).view(x.size(0), -1)
        return 0.5 * self.l2_weighting() * l2_sqnorm


def test_derivs_L2(start_value=0.0, verbose=False):
    R = L2Regularizer(start_value=start_value)
    c = R.l2_weighting(as_float=True)
    dc = R.deriv_l2_weighting(as_float=True)
    X = torch.tensor([[3, 4]], dtype=float)
    Xvec = return_vector(X, as_numpy=True)[0]
    v = torch.ones_like(X)
    vvec = return_vector(v, as_numpy=True)[0]

    # Evaluate R(X) and derivatives
    Rval = R.objfun(X, as_numpy=True)[0][0]
    gX = R.gradient_wrt_inputs(X, as_numpy=True)[0]
    gW = R.gradient_wrt_parameters(X, as_numpy=True)[0]
    Hv = R.hess_vec_wrt_inputs(X, v, as_numpy=True)[0]
    vT_J = R.jac_vec(X, v, as_numpy=True)[0]

    # Calculate expected R(X) and derivatives
    Rval_true = 0.5 * c * np.linalg.norm(Xvec)**2
    gX_true = c * Xvec
    gW_true = 0.5 * np.linalg.norm(Xvec)**2 * dc
    Hv_true = c * vvec  # H = c*Id
    vT_J_true = vvec.T @ Xvec * dc   # J = Xvec * dc, size |x| * |w| = 2*1

    eR = abs(Rval - Rval_true)
    egX = np.linalg.norm(gX - gX_true)
    egW = abs(gW - gW_true)
    eHv = np.linalg.norm(Hv - Hv_true)
    eJ = np.linalg.norm(vT_J - vT_J_true)

    if verbose:
        print("c = %g, c' = %g" % (c, dc))
        print("Rval:", Rval, Rval_true, eR)
        print("gX:", gX, gX_true, egX)
        print("gW:", gW, gW_true, egW)
        print("Hv:", Hv, Hv_true, eHv)
        print("vT_J:", vT_J, vT_J_true, eJ)
    return max(eR, egX, egW, eHv, eJ)


class Conv1d(BaseConvexRegularizer):
    """
    Simple class implementing a Conv1d layer
    """
    def __init__(self, npixels):
        super(Conv1d, self).__init__()
        self.npixels = npixels
        self.n_in_channels = 1
        self.n_out_channels = 1
        self.kernel_size = 3
        self.dilation = 1
        self.stride = 1
        self.padding = 1
        self.padding_mode = 'reflect'
        # self.padding_mode = 'zeros'
        self.has_bias = False
        self.C1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode, bias=self.has_bias, dtype=float)
        self.C2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode, bias=self.has_bias, dtype=float)

    def C_norm(self):
        # Calculate an estimate of ||C||_2
        # For a Conv1D layer without bias, the operator norm is bounded by
        #   sqrt( sum_{Cin, Cout} ||kernel(Cin, Cout)||_1^2 )
        # The kernel information is given in C.weight, tensors of shape (Cout, Cin, kernel_size)
        calc_norm = lambda w: float(torch.linalg.vector_norm(torch.linalg.vector_norm(w, ord=1, dim=2), ord=2))
        return calc_norm(self.C1.weight), calc_norm(self.C2.weight)

    def lip_const(self):
        # Lipschitz constant of gradient of model (wrt inputs x)
        C1_norm, C2_norm = self.C_norm()
        activation_grad_bound = 1.0  # maximum gradient of activation function (SoftPlus)
        activation_grad_lip = 0.25  # Lipschitz constant of gradient of activation function (SoftPlus)

        C1_lip = activation_grad_lip * C1_norm ** 2
        C2_lip = activation_grad_lip * C2_norm ** 2
        # Divide by self.npixels for the averaging layer
        return (C1_lip / self.npixels) + (C2_lip / self.npixels)

    def convex_const(self):
        # Strong convexity constant of model (wrt inputs x)
        return 0.0

    def forward(self, x):
        z1 = torch.nn.functional.softplus(self.C1(x))
        z2 = torch.nn.functional.softplus(self.C2(x))
        # print("z1 =", z1.detach().numpy())
        # print("z2 =", z2.detach().numpy())
        # z has shape (# batches, 2 channels, orig_data_length) - average over orig_data_length
        # Pooling output layer: z_avg[batch,channel] = average of z[batch,channel,...]  # one output per data point and channel
        z1_avg = torch.nn.functional.avg_pool1d(z1, z1.size()[2:]).view(z1.size()[0], -1)
        z2_avg = torch.nn.functional.avg_pool1d(z2, z2.size()[2:]).view(z2.size()[0], -1)
        # print("z1_avg =", z1_avg.detach().numpy())
        # print("z2_avg =", z2_avg.detach().numpy())
        return z1_avg + z2_avg

    def initialize_weights(self, min_val=0.0, max_val=0.001):
        # Initialize random weights before training
        self.C1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        return self

    def project_weights_to_feasible_set(self):
        # Wz and final layer weights must be >= 0 for convexity
        # This function projects weights into this feasible set
        self.C1.weight.data.clamp_(0)
        self.C2.weight.data.clamp_(0)
        return self.parameters()


def test_derivs_conv1d(npixels, min_wt=0.0, max_wt=0.1, verbose=True):
    nchannels = 1
    R = Conv1d(npixels=npixels)
    R.initialize_weights(min_val=min_wt, max_val=max_wt)
    X = torch.from_numpy(np.arange(npixels * nchannels, dtype=float).reshape((1, nchannels, npixels)))

    # Try to replicate calculation
    K1 = R.C1.weight.data.detach().numpy()[0][0]
    K2 = R.C2.weight.data.detach().numpy()[0][0]
    Xvec = X.detach().numpy()[0][0]
    softplus = lambda t: np.log(1 + np.exp(t))
    d_softplus = lambda t: np.exp(t) / (1 + np.exp(t))
    d2_softplus = lambda t: np.exp(t) / (1 + np.exp(t)) ** 2

    def conv1d(v, ker):
        c = np.zeros((len(v),))
        c[0] = (v[1] * ker[0] + v[0] * ker[1] + v[1] * ker[2])
        for i in range(len(v) - 2):
            c[i + 1] = (v[i] * ker[0] + v[i + 1] * ker[1] + v[i + 2] * ker[2])
        c[-1] = (v[-2] * ker[0] + v[-1] * ker[1] + v[-2] * ker[2])
        return c

    def conv1d_3kernel(v, ker):
        # implementation of torch.Conv1d(v) with data 'ker', if len(ker)==3 and padding_mode='reflect'
        # using to test Conv1d class
        assert len(ker) == 3, "Wrong kernel length"
        z = np.zeros((len(v),))
        z[0] = softplus(v[1] * ker[0] + v[0] * ker[1] + v[1] * ker[2])
        for i in range(len(v) - 2):
            z[i + 1] = softplus(v[i] * ker[0] + v[i + 1] * ker[1] + v[i + 2] * ker[2])
        z[-1] = softplus(v[-2] * ker[0] + v[-1] * ker[1] + v[-2] * ker[2])
        # z = softplus(conv1d(v, ker))
        return z

    # gradient of z wrt Xvec
    def gradx_conv1d_3kernel(v, ker):
        assert len(ker) == 3, "Wrong kernel length"
        dz = np.zeros((len(v), len(v)))  # dz[i,j] = dz[i] / dv[j]
        partial_z = d_softplus(conv1d(v, ker))
        # Derivatives of z[0]
        dz[0, 0] = partial_z[0] * ker[1]
        dz[0, 1] = partial_z[0] * (ker[0] + ker[2])
        # Derivatives of z[i]
        for i in range(len(v) - 2):
            dz[i + 1, i] = partial_z[i + 1] * ker[0]
            dz[i + 1, i + 1] = partial_z[i + 1] * ker[1]
            dz[i + 1, i + 2] = partial_z[i + 1] * ker[2]
        # Derivatives of z[-1]
        dz[-1, -2] = partial_z[-1] * (ker[0] + ker[2])
        dz[-1, -1] = partial_z[-1] * ker[1]
        return np.mean(dz, axis=0)

    def gradw_conv1d_3kernel(v, ker):
        assert len(ker) == 3, "Wrong kernel length"
        dz = np.zeros((len(v), len(ker)))  # dz[i,j] = dz[i] / dw[j]
        partial_z = d_softplus(conv1d(v, ker))
        # Derivatives of z[0]
        dz[0, 0] = partial_z[0] * v[1]
        dz[0, 1] = partial_z[0] * v[0]
        dz[0, 2] = partial_z[0] * v[1]
        # Derivatives of z[i]
        for i in range(len(v) - 2):
            dz[i + 1, 0] = partial_z[i + 1] * v[i]
            dz[i + 1, 1] = partial_z[i + 1] * v[i + 1]
            dz[i + 1, 2] = partial_z[i + 1] * v[i + 2]
        # Derivatives of z[-1]
        dz[-1, 0] = partial_z[-1] * v[-2]
        dz[-1, 1] = partial_z[-1] * v[-1]
        dz[-1, 2] = partial_z[-1] * v[-2]
        return np.mean(dz, axis=0)

    def J_conv1d_3kernel(v, ker):
        assert len(ker) == 3, "Wrong kernel length"
        J = np.zeros((len(v), len(ker)))  # J[i, j] = d^2(avg(z)) / dz[i] dw[j]
        partial_z = d_softplus(conv1d(v, ker))
        partial2_z = d2_softplus(conv1d(v, ker))
        # Derivatives of z[0]
        J[1, 0] += partial_z[0] + partial2_z[0] * v[1] * (ker[0] + ker[2])
        J[1, 2] += partial_z[0] + partial2_z[0] * v[1] * (ker[0] + ker[2])
        J[1, 1] += partial2_z[0] * v[0] * (ker[0] + ker[2])
        J[0, 1] += partial_z[0] + partial2_z[0] * v[0] * ker[1]
        J[0, 0] += partial2_z[0] * ker[1] * v[1]
        J[0, 2] += partial2_z[0] * ker[1] * v[1]
        # Derivatives of z[i]
        for i in range(len(v) - 2):
            J[i, 0] += partial_z[i + 1] + partial2_z[i + 1] * v[i] * ker[0]
            J[i, 1] += partial2_z[i + 1] * ker[0] * v[i + 1]
            J[i, 2] += partial2_z[i + 1] * ker[0] * v[i + 2]
            J[i + 1, 1] += partial_z[i + 1] + partial2_z[i + 1] * v[i + 1] * ker[1]
            J[i + 1, 0] += partial2_z[i + 1] * v[i] * ker[1]
            J[i + 1, 2] += partial2_z[i + 1] * v[i + 2] * ker[1]
            J[i + 2, 2] += partial_z[i + 1] + partial2_z[i + 1] * v[i + 2] * ker[2]
            J[i + 2, 0] += partial2_z[i + 1] * v[i] * ker[2]
            J[i + 2, 1] += partial2_z[i + 1] * v[i + 1] * ker[2]
        # Derivatives of z[-1]
        J[-2, 0] += partial_z[-1] + partial2_z[-1] * v[-2] * (ker[0] + ker[2])
        J[-2, 2] += partial_z[-1] + partial2_z[-1] * v[-2] * (ker[0] + ker[2])
        J[-2, 1] += partial2_z[-1] * v[-1] * (ker[0] + ker[2])
        J[-1, 1] += partial_z[-1] + partial2_z[-1] * v[-1] * ker[1]
        J[-1, 0] += partial2_z[-1] * v[-2] * ker[1]
        J[-1, 2] += partial2_z[-1] * v[-2] * ker[1]
        return J / len(v)  # take averages

    def hess_conv1d_3kernel(v, ker):
        assert len(ker) == 3, "Wrong kernel size"
        H = np.zeros((len(v), len(v)))
        partial_z = d_softplus(conv1d(v, ker))
        partial2_z = d2_softplus(conv1d(v, ker))
        # Derivatives of z[0]
        H[0, 0] += partial2_z[0] * ker[1] * ker[1]
        H[0, 1] += partial2_z[0] * (ker[0] + ker[2]) * ker[1]
        H[1, 0] += partial2_z[0] * (ker[0] + ker[2]) * ker[1]
        H[1, 1] += partial2_z[0] * (ker[0] + ker[2]) * (ker[0] + ker[2])
        # Derivatives of z[i]
        for i in range(len(v) - 2):
            H[i:i + 3, i:i + 3] += partial2_z[i + 1] * np.outer(ker, ker)
            # H[i, i] += partial2_z[i+1] * ker[0] * ker[0]
            # H[i, i+1] += partial2_z[i+1] * ker[0] * ker[1]
        # Derivatives of z[-1]
        H[-1, -1] += partial2_z[-1] * ker[1] * ker[1]
        H[-1, -2] += partial2_z[-1] * (ker[0] + ker[2]) * ker[1]
        H[-2, -1] += partial2_z[-1] * (ker[0] + ker[2]) * ker[1]
        H[-2, -2] += partial2_z[-1] * (ker[0] + ker[2]) * (ker[0] + ker[2])
        return H / len(v)  # take averages

    Rval = R.objfun(X, as_numpy=True)[0][0]
    Rval_true = np.mean(conv1d_3kernel(Xvec, K1)) + np.mean(conv1d_3kernel(Xvec, K2))

    gX_true = gradx_conv1d_3kernel(Xvec, K1) + gradx_conv1d_3kernel(Xvec, K2)
    gX = R.gradient_wrt_inputs(X, as_numpy=True)

    gW_true = np.array(list(gradw_conv1d_3kernel(Xvec, K1)) + list(gradw_conv1d_3kernel(Xvec, K2)))
    gW = R.gradient_wrt_parameters(X, as_numpy=True)

    Jtrue = np.hstack((J_conv1d_3kernel(Xvec, K1), J_conv1d_3kernel(Xvec, K2)))
    J = R.jacobian(X, as_numpy=True).reshape(Jtrue.shape)

    Htrue = hess_conv1d_3kernel(Xvec, K1) + hess_conv1d_3kernel(Xvec, K2)
    H = R.hessian(X, as_numpy=True).reshape(Htrue.shape)

    eR = abs(Rval - Rval_true)
    egX = np.linalg.norm(gX - gX_true)
    egW = np.linalg.norm(gW - gW_true)
    eH = np.linalg.norm(H - Htrue)
    eJ = np.linalg.norm(J - Jtrue)

    if verbose:
        print("Rval:", Rval, Rval_true, eR)
        print("gX:", gX, gX_true, egX)
        print("gW:", gW, gW_true, egW)
        print("Hv:", eH)
        print("vT_J:", eJ)
    return max(eR, egX, egW, eH, eJ)


def main():
    print("Test L2")
    for start_value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print(start_value, test_derivs_L2(start_value=start_value, verbose=False))

    print("Test Conv1d")
    for max_wt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(max_wt, test_derivs_conv1d(20, min_wt=0.0, max_wt=max_wt, verbose=False))
    return


if __name__ == '__main__':
    main()
