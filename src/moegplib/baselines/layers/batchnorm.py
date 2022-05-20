import torch
import torch.nn.functional as F
from .layer import VarPropagationLayer


class BatchNorm2DVarPropagationLayer(VarPropagationLayer):
    def __init__(self, layer, use_cov=False, **kwargs):
        super(BatchNorm2DVarPropagationLayer, self).__init__(layer, use_cov, **kwargs)

    def _call_diag_cov(self, x, var):
        # out = x * (self.layer.gamma / (self.layer.moving_variance + self.layer.epsilon))**2
        bz, seq = list(x.size())[:2]
        in_sz = list(x.size())[2:]
        x = x.contiguous().view(bz*seq, *in_sz)
        var = var.contiguous().view(bz*seq, *in_sz)
        
        mean = self.layer(x)
        var = F.batch_norm(var, self.layer.running_mean*0, 
                              self.layer.running_var**2,
                              self.layer.weight**2, 
                              self.layer.bias*0,
                              eps=self.layer.eps)
        return mean.contiguous().view(bz, seq, *in_sz), var.contiguous().view(bz, seq, *in_sz)
