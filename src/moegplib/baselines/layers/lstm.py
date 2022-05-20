import torch
from .layer import VarPropagationLayer
from .layer_utils import get_jacobian_loop 

class LSTMVarPropagationLayer(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(LSTMVarPropagationLayer, self).__init__(layer, use_cov=use_cov, **kwargs)

    def _call_diag_cov(self, x, var):
        nout = self.layer.hidden_size * (2 if self.layer.bidirectional else 1) 
        jcb = get_jacobian_loop(self.layer, x, nout, jcb_bz=1)
        var = torch.einsum("ijkl,ijl->ijk", jcb**2, var)
        mean, _ = self.layer(x)
        return mean, var
