''' Here we store the files that are modified source code of gpytorch.

Additionally, we define patch gp kernels.
'''

import warnings
import torch
import gpytorch
import os
import numpy as np

from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.models.gp import GP
from gpytorch.constraints import Positive
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor
from gpytorch.kernels import Kernel
from copy import deepcopy

from moegplib.moegp.gpytorch import ExactPatchGP


class NeuralTangentKernel(Kernel):
    """ Nerual Tangent Kernel implementation.
    
    Instead of patching the kernels with boundary conditions,
    we introduce "including neighbors" strategy where
    a simple linear kernel can be used as a backbone.
    
    An advantage here is that we can utilize the lazy tensors.
    """
    def __init__(self, num_dimensions=None, offset_prior=None,
                 variance_prior=None, variance_constraint=None, **kwargs):
        super(NeuralTangentKernel, self).__init__(**kwargs)
        
        # setting up the priors
        if num_dimensions is not None:
            warnings.warn("The `num_dimensions` argument is deprecated and no longer used.", DeprecationWarning)
            self.register_parameter(name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)))
        if offset_prior is not None:
            warnings.warn("The `offset_prior` argument is deprecated and no longer used.", DeprecationWarning)
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        if variance_prior is not None:
            self.register_prior("variance_prior", variance_prior, lambda m: m.variance, lambda m, v: m._set_variance(v))
        
        # delta variable should be a positive (indicates the network width)
        if variance_constraint is None:
            variance_constraint = Positive()
        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Forward method of NTK GP.

        Args:
            x1 (tensor): Jacobian of neural network (patched with neighbors)
            x2 (tensor): Jacobian of neural network (patched with neighbors)
            diag (bool, optional): If to return diagonal kernel matrix. Defaults to False.
            last_dim_is_batch (bool, optional): When in batch mode. Defaults to False.

        Returns:
            prod (lazy tensor): The kernel matrix.
        """
        x1_ = x1 * self.variance.sqrt()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        if x1.size() == x2.size() and torch.equal(x1, x2):
            prod = RootLazyTensor(x1_)
        else:
            x2_ = x2 * self.variance.sqrt()
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            prod = MatmulLazyTensor(x1_, x2_.transpose(-2, -1))
        if diag:
            return prod.diag()
        else:
            return prod


class ExactNTKGP(gpytorch.models.ExactGP):
    """
    This model computes the exact NTK Gaussian process.
    Realized with a linear kernel, and train_x & train_y are
    in the transformed spaces.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = NeuralTangentKernel()

    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NTKMISO(ExactPatchGP):
    '''
    Defines NTKGP with multiple input and single output.
    '''
    def __init__(self, jacobian, ytransform, likelihood):
        super(NTKMISO, self).__init__(jacobian, ytransform, likelihood) 
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = PatchNTK() 

    def forward(self, x, **kwargs):
        '''
        Forward function - input to mean and covariance of multivariate normal.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **kwargs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NTK1DGP(ExactPatchGP):
    '''
    Defines NTKGP with linear kernel.
    Jacobian and ytransform are then the inputs.
    Likelihood should be gaussian.
    '''
    def __init__(self, jacobian, ytransform, likelihood):
        super(NTK1DGP, self).__init__(jacobian, ytransform, likelihood) # base class that takes these components
        self.mean_module = gpytorch.means.ConstantMean() # prior mean
        self.covar_module = NeuralTangentKernel() # prior covariance

    def forward(self, x):
        '''
        Forward function - input to mean and covariance of multivariate normal.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)