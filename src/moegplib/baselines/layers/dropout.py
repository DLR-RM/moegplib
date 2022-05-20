import torch

from .layer import VarPropagationLayer


def variance_product_rnd_vars(mean1, mean2, var1, var2):
    return mean1**2*var2 + mean2**2*var1 + var1*var2


def covariance_elementwise_product_rnd_vec(mean1, cov1, mean2, var2):
    """
    Computes covariance element-wise product of one vector (->1)
    with full covariance and another vector with independent elements
    """
    var2_scalar = len(var2.shape) == 0
    var1 = torch.diagonal(cov1, offset=0, dim1=-2, dim2=-1) 
    if var2_scalar:
        term1 = var1 * var2
    else:
        cov2 = torch.diag_embed(var2)
        term1 = torch.mul(cov1, cov2)

    if var2_scalar:
        term2 = mean1 ** 2 * var2
    else:
        term2 = torch.mul(mean1**2, var2)
    term2 = torch.diag_embed(term2)

    # term 3
    if var2_scalar:
        term3 = mean2 ** 2 * cov1
    else:
        mean2_cross = torch.einsum('ij,ik->ijk', mean2, mean2)
        term3 = torch.mul(cov1, mean2_cross)
    return term1 + term2 + term3

class DropoutVarPropagationLayer(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module,
                 initial_noise: bool=False,
                 use_cov: bool=False,
                 **kwargs):
        self.initial_noise = initial_noise
        self.drop_rate = layer.p
        
        super(DropoutVarPropagationLayer, self).__init__(layer, use_cov=use_cov, **kwargs)

    def _call_diag_cov(self, x, in_var):
        if self.initial_noise:
            var = x**2 * self.drop_rate*(1-self.drop_rate)
        else:
            new_mean = 1 - self.drop_rate
            new_var = self.drop_rate * (1 - self.drop_rate)
            mean = x 
            var = variance_product_rnd_vars(mean, new_mean, in_var, new_var)/(1 - self.drop_rate)**2
        return self.layer(x), var

    def _call_full_cov(self, x, var):
        if self.initial_noise:
            var = x**2 * self.drop_rate*(1-self.drop_rate)
            var = torch.diag_embed(var)
        else:
            mean = x  
            mean_shape = mean.shape
            new_mean = torch.ones(mean_shape, dtype=torch.float) * (1 - self.drop_rate)
            new_var = torch.ones(mean_shape, dtype=torch.float) * self.drop_rate * (1 - self.drop_rate)
            if self.cuda:
                new_mean = new_mean.cuda()
                new_var = new_var.cuda()            
            var = covariance_elementwise_product_rnd_vec(mean, in_var, new_mean, new_var)
        return self.layer(x), var
