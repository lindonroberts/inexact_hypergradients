#!/usr/bin/env python3

"""
A collection of neural net regularizers
"""
import torch
from torch.nn.functional import softplus, avg_pool1d
try:
    from ad_testing_v2.utils import BaseConvexRegularizer, AbstractDataset
except ModuleNotFoundError:
    from utils import BaseConvexRegularizer, AbstractDataset

__all__ = ['ShallowConv1d', 'DeepConv1d', 'VDeepConv1d']


class ShallowConv1d(BaseConvexRegularizer):
    """
    Simple class implementing a Conv1d layer plus L2 regularizer, on top of regular denoising
    """

    def __init__(self, dataset: AbstractDataset, beta=100.0, conv_start_value=0.0, L2_start_value=-5.0,
                 have_fixed_beta=True, have_fixed_L2=False, symmetric_kernel=False, keep_tv_kernel=False):
        super(ShallowConv1d, self).__init__(dataset)
        self.npixels = self.dataset.npixels

        # Conv1d layer
        self.have_fixed_beta = have_fixed_beta
        assert self.have_fixed_beta, "Only fixed beta supported for now"
        if not self.have_fixed_beta:  # parameter for softplus
            self.beta = torch.nn.Parameter((beta) * torch.ones(1))
        else:
            self.beta = float(beta)

        self.n_in_channels = 1
        self.n_out_channels = 1
        self.kernel_size = 3
        self.dilation = 1
        self.stride = 1
        self.padding = 1
        self.padding_mode = 'reflect'
        # self.padding_mode = 'zeros'
        self.has_bias = False
        self.symmetric_kernel = symmetric_kernel  # if true, C2 = (-1)*C1 always
        self.keep_tv_kernel = keep_tv_kernel  # only learn weights, not kernel?
        self.C1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=self.has_bias, dtype=float)
        if not self.symmetric_kernel:
            self.C2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                      padding_mode=self.padding_mode, bias=self.has_bias, dtype=float)
        else:
            self.C2 = None
        self.conv_nn_weight = torch.nn.Parameter((conv_start_value) * torch.ones(1))

        # L2 weight
        self.have_fixed_L2 = have_fixed_L2
        if not self.have_fixed_L2:
            self.L2_penalty_param = torch.nn.Parameter((L2_start_value) * torch.ones(1))
        else:
            self.L2_penalty_param = torch.tensor(L2_start_value)

    def l2_weighting(self, as_float=False):  # c(self.l2_penalty)
        c = softplus(self.L2_penalty_param)
        return float(c) if as_float else c

    def conv_nn_weighting(self, as_float=False):
        c = softplus(self.conv_nn_weight)
        return float(c) if as_float else c

    def C_norm(self):
        # Calculate an estimate of ||C||_2
        # For a Conv1D layer without bias, the operator norm is bounded by
        #   sqrt( sum_{Cin, Cout} ||kernel(Cin, Cout)||_1^2 )
        # The kernel information is given in C.weight, tensors of shape (Cout, Cin, kernel_size)
        calc_norm = lambda w: float(torch.linalg.vector_norm(torch.linalg.vector_norm(w, ord=1, dim=2), ord=2))
        C1_norm = calc_norm(self.C1.weight)
        if self.symmetric_kernel:
            C2_norm = C1_norm
        else:
            C2_norm = calc_norm(self.C2.weight)
        return C1_norm, C2_norm

    def lip_const(self):
        Knorm = self.dataset.fwd_op_norm

        # Lipschitz constant of gradient of model (wrt inputs x)
        C1_norm, C2_norm = self.C_norm()
        # softplus(t, beta) = 1/beta * log(1 + exp(beta*t))
        # d/dt softplus(t, beta) = exp(beta*t) / (exp(beta*t) + 1)
        # d^2/dt^2 softplus(t, beta) = beta * exp(beta*t) / (exp(beta*t)+1)^2
        # Second derivative is maximized at t=0, with value beta/4
        activation_grad_bound = 1.0  # maximum gradient of activation function (SoftPlus)
        activation_grad_lip = 0.25 * self.beta  # Lipschitz constant of gradient of activation function (SoftPlus)

        C1_lip = float(activation_grad_lip * C1_norm ** 2)
        C2_lip = float(activation_grad_lip * C2_norm ** 2)
        # Divide by self.npixels for the averaging layer
        return Knorm**2 + self.conv_nn_weighting(as_float=True) * (C1_lip + C2_lip) / self.npixels + self.l2_weighting(as_float=True)

    def convex_const(self):
        # Strong convexity constant of model (wrt inputs x)
        return 0.0 + self.l2_weighting(as_float=True)

    def forward(self, x):
        data_sqloss = self.data_sq_loss(x)  # ||fwd_op(x) - noisy_data[self.current_index]||^2

        z1 = softplus(self.C1(x), beta=self.beta)
        if self.symmetric_kernel:
            z2 = softplus(-self.C1(x), beta=self.beta)
        else:
            z2 = softplus(self.C2(x), beta=self.beta)

        # Pooling output layer: z_avg[batch,channel] = average of z[batch,channel,...]  # one output per data point and channel
        z1_avg = avg_pool1d(z1, z1.size()[2:]).view(z1.size()[0], -1)
        z2_avg = avg_pool1d(z2, z2.size()[2:]).view(z2.size()[0], -1)

        l2_sqnorm = torch.sum(x.view(x.size(0), -1) ** 2, dim=1).view(x.size(0), -1)
        return data_sqloss + self.conv_nn_weighting() * (z1_avg + z2_avg) + 0.5 * self.l2_weighting() * l2_sqnorm

    def initialize_weights_random(self, min_val=-1.0, max_val=1.0):
        # Initialize random weights before training
        self.C1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        if not self.symmetric_kernel:
            self.C2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        return self

    def initialize_weights_tv(self):
        self.C1.weight.data[0, 0, 0] = 1.0
        self.C1.weight.data[0, 0, 1] = -1.0
        self.C1.weight.data[0, 0, 2] = 0.0
        if not self.symmetric_kernel:
            self.C2.weight.data[0, 0, 0] = -1.0
            self.C2.weight.data[0, 0, 1] = 1.0
            self.C2.weight.data[0, 0, 2] = 0.0
        return self

    def project_weights_to_feasible_set(self):
        # Wz and final layer weights must be >= 0 for convexity
        if self.keep_tv_kernel:
            self.initialize_weights_tv()
        return self.parameters()


class DeepConv1d(BaseConvexRegularizer):
    """
    Simple class implementing a Conv1d layer plus L2 regularizer, on top of regular denoising
    """

    def __init__(self, dataset: AbstractDataset, beta=100.0, conv_start_value=0.0, L2_start_value=-5.0,
                 have_fixed_beta=True, have_fixed_L2=False):
        # Depth = # hidden layers (depth=0 corresponds to ShallowConv1d class)
        super(DeepConv1d, self).__init__(dataset)
        self.npixels = self.dataset.npixels

        # Conv1d layer
        self.have_fixed_beta = have_fixed_beta
        assert self.have_fixed_beta, "Only fixed beta supported for now"
        if not self.have_fixed_beta:  # parameter for softplus
            self.beta = torch.nn.Parameter((beta) * torch.ones(1))
        else:
            self.beta = float(beta)

        self.n_in_channels = 1
        self.n_out_channels = 1
        self.kernel_size = 3
        self.dilation = 1
        self.stride = 1
        self.padding = 1
        self.padding_mode = 'reflect'
        # self.padding_mode = 'zeros'

        # First layer (with bias)
        self.C1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)

        # Second layer (Cn_x have bias)
        self.C3_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C3_z1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C3_z2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        self.C4_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C4_z1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C4_z2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        self.conv_nn_weight = torch.nn.Parameter((conv_start_value) * torch.ones(1))

        # L2 weight
        self.have_fixed_L2 = have_fixed_L2
        if not self.have_fixed_L2:
            self.L2_penalty_param = torch.nn.Parameter((L2_start_value) * torch.ones(1))
        else:
            self.L2_penalty_param = torch.tensor(L2_start_value)

    def l2_weighting(self, as_float=False):  # c(self.l2_penalty)
        c = softplus(self.L2_penalty_param)
        return float(c) if as_float else c

    def conv_nn_weighting(self, as_float=False):
        c = softplus(self.conv_nn_weight)
        return float(c) if as_float else c

    def C_norm(self):
        # Calculate an estimate of ||C||_2
        # For a Conv1D layer without bias, the operator norm is bounded by
        #   sqrt( sum_{Cin, Cout} ||kernel(Cin, Cout)||_1^2 )
        # The kernel information is given in C.weight, tensors of shape (Cout, Cin, kernel_size)
        calc_norm = lambda w: float(torch.linalg.vector_norm(torch.linalg.vector_norm(w, ord=1, dim=2), ord=2))

        C_norms = {}
        C_norms['C1'] = calc_norm(self.C1.weight)
        C_norms['C2'] = calc_norm(self.C2.weight)
        C_norms['C3_x'] = calc_norm(self.C3_x.weight)
        C_norms['C3_z1'] = calc_norm(self.C3_z1.weight)
        C_norms['C3_z2'] = calc_norm(self.C3_z2.weight)
        C_norms['C4_x'] = calc_norm(self.C4_x.weight)
        C_norms['C4_z1'] = calc_norm(self.C4_z1.weight)
        C_norms['C4_z2'] = calc_norm(self.C4_z2.weight)

        return C_norms

    def lip_const(self):
        Knorm = self.dataset.fwd_op_norm

        # Lipschitz constant of gradient of model (wrt inputs x)
        C_norms = self.C_norm()
        # softplus(t, beta) = 1/beta * log(1 + exp(beta*t))
        # d/dt softplus(t, beta) = exp(beta*t) / (exp(beta*t) + 1)
        # d^2/dt^2 softplus(t, beta) = beta * exp(beta*t) / (exp(beta*t)+1)^2
        # Second derivative is maximized at t=0, with value beta/4
        activation_grad_bound = 1.0  # maximum gradient of activation function (SoftPlus)
        activation_grad_lip = 0.25 * float(self.beta)  # Lipschitz constant of gradient of activation function (SoftPlus)

        z1_lip = activation_grad_lip * C_norms['C1'] ** 2
        z1_grad_bound = activation_grad_bound * C_norms['C1']
        z2_lip = activation_grad_lip * C_norms['C2'] ** 2
        z2_grad_bound = activation_grad_bound * C_norms['C2']

        # z3 = softplus(y3), bound on ||y3(x1)-y3(x2)|| <= C * ||x1-x2||
        y3_lip = C_norms['C3_x'] + C_norms['C3_z1'] * z1_grad_bound + C_norms['C3_z2'] * z2_grad_bound
        z3_lip = activation_grad_lip * y3_lip * C_norms['C3_x'] \
                 + C_norms['C3_z1'] * z1_grad_bound * activation_grad_lip * y3_lip \
                 + activation_grad_bound * C_norms['C3_z1'] * z1_lip \
                 + C_norms['C3_z2'] * z2_grad_bound * activation_grad_lip * y3_lip \
                 + activation_grad_bound * C_norms['C3_z2'] * z2_lip
        z3_grad_bound = activation_grad_bound * C_norms['C3_x'] \
                        + activation_grad_bound * C_norms['C3_z1'] * z1_grad_bound \
                        + activation_grad_bound * C_norms['C3_z2'] * z2_grad_bound

        # And everything the same again for z4
        y4_lip = C_norms['C4_x'] + C_norms['C4_z1'] * z1_grad_bound + C_norms['C4_z2'] * z2_grad_bound
        z4_lip = activation_grad_lip * y4_lip * C_norms['C4_x'] \
                 + C_norms['C4_z1'] * z1_grad_bound * activation_grad_lip * y4_lip \
                 + activation_grad_bound * C_norms['C4_z1'] * z1_lip \
                 + C_norms['C4_z2'] * z2_grad_bound * activation_grad_lip * y4_lip \
                 + activation_grad_bound * C_norms['C4_z2'] * z2_lip
        z4_grad_bound = activation_grad_bound * C_norms['C4_x'] \
                        + activation_grad_bound * C_norms['C4_z1'] * z1_grad_bound \
                        + activation_grad_bound * C_norms['C4_z2'] * z2_grad_bound

        # Divide by self.npixels for the averaging layer
        return Knorm**2 + self.conv_nn_weighting(as_float=True) * (z3_lip + z4_lip) / self.npixels + self.l2_weighting(as_float=True)

    def convex_const(self):
        # Strong convexity constant of model (wrt inputs x)
        return 0.0 + self.l2_weighting(as_float=True)

    def forward(self, x):
        # Pooling output layer: z_avg[batch,channel] = average of z[batch,channel,...]  # one output per data point and channel
        avg = lambda z: avg_pool1d(z, z.size()[2:]).view(z.size()[0], -1)

        data_sqloss = self.data_sq_loss(x)  # ||fwd_op(x) - noisy_data[self.current_index]||^2

        # First layer
        z1 = softplus(self.C1(x), beta=self.beta)
        z2 = softplus(self.C2(x), beta=self.beta)

        # Second layer
        z3 = softplus(self.C3_x(x) + self.C3_z1(z1) + self.C3_z2(z2), beta=self.beta)
        z4 = softplus(self.C4_x(x) + self.C4_z1(z1) + self.C4_z2(z2), beta=self.beta)

        z3_avg = avg(z3)
        z4_avg = avg(z4)

        l2_sqnorm = torch.sum(x.view(x.size(0), -1) ** 2, dim=1).view(x.size(0), -1)
        return data_sqloss + self.conv_nn_weighting() * (z3_avg + z4_avg) + 0.5 * self.l2_weighting() * l2_sqnorm

    def initialize_weights_random(self, min_val=-1.0, max_val=1.0):
        # Initialize random weights before training
        self.C1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_z1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_z2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_z1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_z2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        return self

    def project_weights_to_feasible_set(self):
        # Wz and final layer weights must be >= 0 for convexity
        self.C3_z1.weight.data.clamp_(0)
        self.C3_z2.weight.data.clamp_(0)
        self.C4_z1.weight.data.clamp_(0)
        self.C4_z2.weight.data.clamp_(0)
        return self.parameters()


class VDeepConv1d(BaseConvexRegularizer):
    """
    Simple class implementing a Conv1d layer plus L2 regularizer, on top of regular denoising
    """

    def __init__(self, dataset: AbstractDataset, beta=100.0, conv_start_value=0.0, L2_start_value=-5.0,
                 have_fixed_beta=True, have_fixed_L2=False):
        # Depth = # hidden layers (depth=0 corresponds to ShallowConv1d class)
        super(VDeepConv1d, self).__init__(dataset)
        self.npixels = self.dataset.npixels

        # Conv1d layer
        self.have_fixed_beta = have_fixed_beta
        assert self.have_fixed_beta, "Only fixed beta supported for now"
        if not self.have_fixed_beta:  # parameter for softplus
            self.beta = torch.nn.Parameter((beta) * torch.ones(1))
        else:
            self.beta = float(beta)

        self.n_in_channels = 1
        self.n_out_channels = 1
        self.kernel_size = 3
        self.dilation = 1
        self.stride = 1
        self.padding = 1
        self.padding_mode = 'reflect'
        # self.padding_mode = 'zeros'

        # First layer (with bias)
        self.C1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)

        # Second layer (Cn_x have bias)
        self.C3_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C3_z1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                  kernel_size=self.kernel_size,  stride=self.stride, padding=self.padding,
                                  padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C3_z2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        self.C4_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C4_z1 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C4_z2 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        # Third layer (Cn_x have bias)
        self.C5_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C5_z3 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C5_z4 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        self.C6_x = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    padding_mode=self.padding_mode, bias=True, dtype=float)
        self.C6_z3 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)
        self.C6_z4 = torch.nn.Conv1d(self.n_in_channels, self.n_out_channels, dilation=self.dilation,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                     padding_mode=self.padding_mode, bias=False, dtype=float)

        self.conv_nn_weight = torch.nn.Parameter((conv_start_value) * torch.ones(1))

        # L2 weight
        self.have_fixed_L2 = have_fixed_L2
        if not self.have_fixed_L2:
            self.L2_penalty_param = torch.nn.Parameter((L2_start_value) * torch.ones(1))
        else:
            self.L2_penalty_param = torch.tensor(L2_start_value)

    def l2_weighting(self, as_float=False):  # c(self.l2_penalty)
        c = softplus(self.L2_penalty_param)
        return float(c) if as_float else c

    def conv_nn_weighting(self, as_float=False):
        c = softplus(self.conv_nn_weight)
        return float(c) if as_float else c

    def C_norm(self):
        # Calculate an estimate of ||C||_2
        # For a Conv1D layer without bias, the operator norm is bounded by
        #   sqrt( sum_{Cin, Cout} ||kernel(Cin, Cout)||_1^2 )
        # The kernel information is given in C.weight, tensors of shape (Cout, Cin, kernel_size)
        calc_norm = lambda w: float(torch.linalg.vector_norm(torch.linalg.vector_norm(w, ord=1, dim=2), ord=2))

        C_norms = {}
        C_norms['C1'] = calc_norm(self.C1.weight)
        C_norms['C2'] = calc_norm(self.C2.weight)
        C_norms['C3_x'] = calc_norm(self.C3_x.weight)
        C_norms['C3_z1'] = calc_norm(self.C3_z1.weight)
        C_norms['C3_z2'] = calc_norm(self.C3_z2.weight)
        C_norms['C4_x'] = calc_norm(self.C4_x.weight)
        C_norms['C4_z1'] = calc_norm(self.C4_z1.weight)
        C_norms['C4_z2'] = calc_norm(self.C4_z2.weight)
        C_norms['C5_x'] = calc_norm(self.C5_x.weight)
        C_norms['C5_z3'] = calc_norm(self.C5_z3.weight)
        C_norms['C5_z4'] = calc_norm(self.C5_z4.weight)
        C_norms['C6_x'] = calc_norm(self.C6_x.weight)
        C_norms['C6_z3'] = calc_norm(self.C6_z3.weight)
        C_norms['C6_z4'] = calc_norm(self.C6_z4.weight)

        return C_norms

    def lip_const(self):
        Knorm = self.dataset.fwd_op_norm

        # Lipschitz constant of gradient of model (wrt inputs x)
        C_norms = self.C_norm()
        # softplus(t, beta) = 1/beta * log(1 + exp(beta*t))
        # d/dt softplus(t, beta) = exp(beta*t) / (exp(beta*t) + 1)
        # d^2/dt^2 softplus(t, beta) = beta * exp(beta*t) / (exp(beta*t)+1)^2
        # Second derivative is maximized at t=0, with value beta/4
        activation_grad_bound = 1.0  # maximum gradient of activation function (SoftPlus)
        activation_grad_lip = 0.25 * float(self.beta)  # Lipschitz constant of gradient of activation function (SoftPlus)

        z1_lip = activation_grad_lip * C_norms['C1'] ** 2
        z1_grad_bound = activation_grad_bound * C_norms['C1']
        z2_lip = activation_grad_lip * C_norms['C2'] ** 2
        z2_grad_bound = activation_grad_bound * C_norms['C2']

        # z3 = softplus(y3), bound on ||y3(x1)-y3(x2)|| <= C * ||x1-x2||
        y3_lip = C_norms['C3_x'] + C_norms['C3_z1'] * z1_grad_bound + C_norms['C3_z2'] * z2_grad_bound
        z3_lip = activation_grad_lip * y3_lip * C_norms['C3_x'] \
                 + C_norms['C3_z1'] * z1_grad_bound * activation_grad_lip * y3_lip \
                 + activation_grad_bound * C_norms['C3_z1'] * z1_lip \
                 + C_norms['C3_z2'] * z2_grad_bound * activation_grad_lip * y3_lip \
                 + activation_grad_bound * C_norms['C3_z2'] * z2_lip
        z3_grad_bound = activation_grad_bound * C_norms['C3_x'] \
                        + activation_grad_bound * C_norms['C3_z1'] * z1_grad_bound \
                        + activation_grad_bound * C_norms['C3_z2'] * z2_grad_bound

        # And everything the same again for z4
        y4_lip = C_norms['C4_x'] + C_norms['C4_z1'] * z1_grad_bound + C_norms['C4_z2'] * z2_grad_bound
        z4_lip = activation_grad_lip * y4_lip * C_norms['C4_x'] \
                 + C_norms['C4_z1'] * z1_grad_bound * activation_grad_lip * y4_lip \
                 + activation_grad_bound * C_norms['C4_z1'] * z1_lip \
                 + C_norms['C4_z2'] * z2_grad_bound * activation_grad_lip * y4_lip \
                 + activation_grad_bound * C_norms['C4_z2'] * z2_lip
        z4_grad_bound = activation_grad_bound * C_norms['C4_x'] \
                        + activation_grad_bound * C_norms['C4_z1'] * z1_grad_bound \
                        + activation_grad_bound * C_norms['C4_z2'] * z2_grad_bound

        # z5 = softplus(y5), bound on ||y3(x1)-y3(x2)|| <= C * ||x1-x2||
        y5_lip = C_norms['C5_x'] + C_norms['C5_z3'] * z3_grad_bound + C_norms['C5_z4'] * z4_grad_bound
        z5_lip = activation_grad_lip * y5_lip * C_norms['C5_x'] \
                 + C_norms['C5_z3'] * z3_grad_bound * activation_grad_lip * y5_lip \
                 + activation_grad_bound * C_norms['C5_z3'] * z3_lip \
                 + C_norms['C5_z4'] * z4_grad_bound * activation_grad_lip * y5_lip \
                 + activation_grad_bound * C_norms['C5_z4'] * z4_lip
        z5_grad_bound = activation_grad_bound * C_norms['C5_x'] \
                        + activation_grad_bound * C_norms['C5_z3'] * z3_grad_bound \
                        + activation_grad_bound * C_norms['C5_z4'] * z4_grad_bound

        # And everything the same again for z6
        y6_lip = C_norms['C6_x'] + C_norms['C6_z3'] * z3_grad_bound + C_norms['C6_z4'] * z4_grad_bound
        z6_lip = activation_grad_lip * y6_lip * C_norms['C6_x'] \
                 + C_norms['C6_z3'] * z3_grad_bound * activation_grad_lip * y6_lip \
                 + activation_grad_bound * C_norms['C6_z3'] * z3_lip \
                 + C_norms['C6_z4'] * z4_grad_bound * activation_grad_lip * y6_lip \
                 + activation_grad_bound * C_norms['C6_z4'] * z4_lip
        z6_grad_bound = activation_grad_bound * C_norms['C6_x'] \
                        + activation_grad_bound * C_norms['C6_z3'] * z3_grad_bound \
                        + activation_grad_bound * C_norms['C6_z4'] * z4_grad_bound

        # Divide by self.npixels for the averaging layer
        return Knorm**2 + self.conv_nn_weighting(as_float=True) * (z5_lip + z6_lip) / self.npixels + self.l2_weighting(as_float=True)

    def convex_const(self):
        # Strong convexity constant of model (wrt inputs x)
        return 0.0 + self.l2_weighting(as_float=True)

    def forward(self, x):
        # Pooling output layer: z_avg[batch,channel] = average of z[batch,channel,...]  # one output per data point and channel
        avg = lambda z: avg_pool1d(z, z.size()[2:]).view(z.size()[0], -1)

        data_sqloss = self.data_sq_loss(x)  # ||fwd_op(x) - noisy_data[self.current_index]||^2

        # First layer
        z1 = softplus(self.C1(x), beta=self.beta)
        z2 = softplus(self.C2(x), beta=self.beta)

        # Second layer
        z3 = softplus(self.C3_x(x) + self.C3_z1(z1) + self.C3_z2(z2), beta=self.beta)
        z4 = softplus(self.C4_x(x) + self.C4_z1(z1) + self.C4_z2(z2), beta=self.beta)

        # Third layer
        z5 = softplus(self.C5_x(x) + self.C5_z3(z3) + self.C5_z4(z4), beta=self.beta)
        z6 = softplus(self.C6_x(x) + self.C6_z3(z3) + self.C6_z4(z4), beta=self.beta)

        z5_avg = avg(z5)
        z6_avg = avg(z6)

        l2_sqnorm = torch.sum(x.view(x.size(0), -1) ** 2, dim=1).view(x.size(0), -1)
        return data_sqloss + self.conv_nn_weighting() * (z5_avg + z6_avg) + 0.5 * self.l2_weighting() * l2_sqnorm

    def initialize_weights_random(self, min_val=-1.0, max_val=1.0):
        # Initialize random weights before training
        self.C1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_z1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C3_z2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_z1.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C4_z2.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C5_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C5_z3.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C5_z4.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C6_x.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C6_z3.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        self.C6_z4.weight.data = min_val + (max_val - min_val) * torch.rand(self.n_out_channels, self.n_in_channels, self.kernel_size).double()
        return self

    def initialize_weights_good(self):
        # Initialize weights using a known good combination
        # End result of starting from random weights after 2.4m lower-level iters with alpha=1e-1 and tol=1e-2
        good_param_vec = torch.tensor([-1.380639672, -8.358563423, 0.345899821, 0.164801922, -0.238285199, 0.363557201,
                                       -1.006677874, 0.503031451, 0.494075242, -0.900703473, 1.779351747, 0.21578105,
                                       0.379965126, 0.869139709, 0, 0, 0, 0.751397858, 0.625220735, 0.850153422,
                                       1.732280543, 0.769672697, -1.26162589, 0.654863962, 0, 0, 0, 1.044858269,
                                       0.153390632, 0.164420999, -1.994468213, 4.947192125, -3.061585539, -0.081747801,
                                       0.118775705, 0.103989267, 0.09626504, 0.075690451, 0.112401729, 0.092839957,
                                       -2.77621327, -3.010891944, 4.970484328, -0.679789478, 0.006371228, 0.002700298,
                                       0.017936468, 0.001550439, 0.00712218, 0.008702088])
        self.set_params(good_param_vec)
        return

    def initialize_weights_tv(self):
        self.C1.weight.data[0, 0, :] = 0.0
        self.C2.weight.data[0, 0, :] = 0.0
        self.C3_x.weight.data[0, 0, :] = 0.0
        self.C3_z1.weight.data[0, 0, :] = 0.0
        self.C3_z2.weight.data[0, 0, :] = 0.0
        self.C4_x.weight.data[0, 0, :] = 0.0
        self.C4_z1.weight.data[0, 0, :] = 0.0
        self.C4_z2.weight.data[0, 0, :] = 0.0
        # self.C5_x.weight.data[0, 0, :] = 0.0
        self.C5_z3.weight.data[0, 0, :] = 0.0
        self.C5_z4.weight.data[0, 0, :] = 0.0
        # self.C6_x.weight.data[0, 0, :] = 0.0
        self.C6_z3.weight.data[0, 0, :] = 0.0
        self.C6_z4.weight.data[0, 0, :] = 0.0
        self.C5_x.weight.data[0, 0, 0] = 1.0
        self.C5_x.weight.data[0, 0, 1] = -1.0
        self.C5_x.weight.data[0, 0, 2] = 0.0
        self.C6_x.weight.data[0, 0, 0] = -1.0
        self.C6_x.weight.data[0, 0, 1] = 1.0
        self.C6_x.weight.data[0, 0, 2] = 0.0
        return self

    def project_weights_to_feasible_set(self):
        # Wz and final layer weights must be >= 0 for convexity
        self.C3_z1.weight.data.clamp_(0)
        self.C3_z2.weight.data.clamp_(0)
        self.C4_z1.weight.data.clamp_(0)
        self.C4_z2.weight.data.clamp_(0)
        self.C5_z3.weight.data.clamp_(0)
        self.C5_z4.weight.data.clamp_(0)
        self.C6_z3.weight.data.clamp_(0)
        self.C6_z4.weight.data.clamp_(0)
        return self.parameters()