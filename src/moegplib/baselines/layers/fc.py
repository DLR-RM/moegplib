import torch
from .layer import VarPropagationLayer
from .layer_utils import get_jacobian_loop

class DenseVarPropagationLayer(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(DenseVarPropagationLayer, self).__init__(layer, use_cov, **kwargs)
        self.weights = layer.weight

    def _call_diag_cov(self, x, var):
        x = self.layer(x)
        var = torch.matmul(var, (self.weights**2.).t())
        return x, var

    def _call_full_cov(self, x, var):
        return self.layer(x), torch.tensordot(torch.tensordot(var, self.weights, dims=([2], [1])), 
                               self.weights, dims=([1], [1]))
        
class DenseVarPropagationLayer2(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(DenseVarPropagationLayer, self).__init__(layer, use_cov, **kwargs)
        self.weights = layer.weight 

    def _call_diag_cov(self, x, var):
        # var: [bz, seq, in]
        nout = self.layer.out_features 
        jcb = get_jacobian_loop(self.layer, x, nout) 
        var = torch.matmul(jcb ** 2, var) 
        return self.layer(x), var

    def _call_full_cov(self, x, var):
        return self.layer(x), torch.tensordot(torch.tensordot(var, self.weights, dims=([2], [1])), 
                               self.weights, dims=([1], [1]))
