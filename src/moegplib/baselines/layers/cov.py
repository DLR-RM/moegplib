import torch
import torch.nn.functional as F
from .layer import VarPropagationLayer
from .layer_utils import get_jacobian_loop


class Conv2DVarPropagationLayer(VarPropagationLayer):

    def __init__(self, layer: torch.nn.Module, use_cov=False, **kwargs):
        super(Conv2DVarPropagationLayer, self).__init__(layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov(self, x, var):
        
        bz, seq = list(x.size())[:2]
        in_sz = list(x.size())[2:]
        x = x.contiguous().view(bz*seq, *in_sz)
        var = var.contiguous().view(bz*seq, *in_sz)

        oc = self.layer.in_channels
        ic = self.layer.out_channels
        kernel = self.layer.weight
        pad = self.layer.padding
        stride = self.layer.stride
        
        mean = self.layer(x)
        out_sz = list(mean.size())[1:]
        var = F.conv2d(var, kernel**2, padding=pad, stride=stride)
        return mean.contiguous().view(bz, seq, *out_sz), var.contiguous().view(bz, seq, *out_sz)

