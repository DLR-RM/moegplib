import torch
from .layer import VarPropagationLayer
from .layer_utils import get_jacobian_loop 

class MHAVarPropagationLayer(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(MHAVarPropagationLayer, self).__init__(layer, use_cov=use_cov, **kwargs)

    def _call_diag_cov(self, x, var):
        nout = self.layer.embed_dim 
        jcb = get_jacobian_loop(self.layer, x, nout, jcb_bz=1)
        var = torch.einsum("ijkl,ijl->ijk", jcb**2, var)
        x = x.permute(1, 0, 2)
        mean, _ = self.layer(x, x, x)
        return mean.permute(1, 0, 2), var
