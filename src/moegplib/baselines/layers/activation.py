import torch
import torch.nn.functional as F

from .layer import ActivationVarPropagationLayer    

def relu_var_tf(mean, var, eps=1e-8):
    std = torch.sqrt(var+eps)
    exp = mean/(torch.sqrt(2.0)*std)
    erf_exp = torch.erf(exp)
    exp_exp2 = torch.exp(-1*exp**2)
    term1 = 0.5 * (var+mean**2) * (erf_exp + 1)
    term2 = mean*std/(torch.sqrt(2*math.pi))*exp_exp2
    term3 = mean/2*(1+erf_exp)
    term4 = torch.sqrt(1/2/math.pi)*std*exp_exp2
    return F.relu(term1 + term2 - (term3 + term4)**2)

def jacobian_tanh_tf(x):
    """only diagonal of J needed bc tanh is applied element-wise
    
    args:
        x: tensor, input to the tanh activation function
    """
    return 1 - torch.tanh(x)**2

class SigmoidActivationVarPropagationLayer(ActivationVarPropagationLayer):
    def __init__(self, layer, use_cov=False, **kwargs):
        super(SigmoidActivationVarPropagationLayer, self).__init__(layer=layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov_approx(self, x, var):
        y = self.layer(x)
        dsigmoid = torch.mul(y, 1 - y)
        var = torch.mul(var, dsigmoid ** 2)
        return y, var

class LinearActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Linear activation function. Identity...
    """

    def __init__(self, layer=None, use_cov=False, **kwargs):
        super(LinearActivationVarPropagationLayer, self).__init__(layer=layer, use_cov=use_cov, **kwargs)

    def call(self, x, var):
        return self.layer(x), var

class LeakyReLUActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Variance propagation through LeakyReLU activation function.
    """

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(LeakyReLUActivationVarPropagationLayer, self).__init__(layer=layer, use_cov=use_cov, **kwargs)
        
    def _call_full_cov_approx(self, x, var):
        pass

    def _call_diag_cov_exact(self, x, var):
        """
        approximate propagation with diagonal covariance matrix
        """
        return self.layer(x), relu_var_tf(x, var)
    
    def _call_diag_cov_approx(self, x, var):
        jacobian = ((x > 0) + self.layer.negative_slope ** 2)   
        # print("out_mean size of leaky relu: ", self.layer(x).size()))
        return self.layer(x), torch.mul(var, jacobian.float())

class ReLUActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Variance propagation through ReLU activation function.
    """

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(ReLUActivationVarPropagationLayer, self).__init__(layer=layer, use_cov=use_cov, **kwargs)
        
    def _call_full_cov_approx(self, x, var):
        """
        approximate propagation of full covariance matrix
        """
        # tf.multiply (element-wise mul), tf.to_float(self.inputs > 0)
        # print((self.input > 0))
        return self.layer(x), torch.mul(var, 
                        torch.einsum('ij,ik->ijk', (self.input > 0).float(), (self.input > 0).float())) 

    def _call_diag_cov_exact(self, x, var):
        """
        approximate propagation with diagonal covariance matrix
        """
        return self.layer(x), relu_var_tf(x, var)
    
    def _call_diag_cov_approx(self, x, var):
        """
        approximate propagation with diagonal covariance matrix and exact variance computation
        under gaussian assumption
        """
        return self.layer(x), torch.mul(var, (x > 0).float())

    
class TanhActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Variance propagation through ReLU activation function.
    """

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(TanhActivationVarPropagationLayer, self).__init__(layer=layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov_approx(self, x, var):
        """
        approximate propagation with diagonal covariance matrix and exact variance computation
        under gaussian assumption
        """
        J = jacobian_tanh_tf(x)
        return self.layer(x), torch.mul(var, J**2)    
