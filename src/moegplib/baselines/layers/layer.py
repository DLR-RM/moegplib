import torch
import time 

class VarPropagationLayer(torch.nn.Module):
    """
    General layer for variance propagation
    
    properties:
        layer: torch nn.Module
        use_cov: bool, use full covariance or not
    """
    
    def __init__(self,
                 layer: torch.nn.Module,
                 use_cov: bool=False,
                 **kwargs):
        super(VarPropagationLayer, self).__init__()
        # print(kwargs)
        self.layer = layer
        self.use_cov = use_cov
        self.cuda = kwargs['cuda']

    def forward(self, x, var):
        """
        call method using full covariance or diagonal covariance
        """
        m_name = self.layer.__class__.__name__
        if var is not None or m_name == "Dropout":
            if self.use_cov:
                mean, var = self._call_full_cov(x, var)
            else:
                mean, var = self._call_diag_cov(x, var)
            return mean, var
        else:
            if m_name in {"Conv2d", "BatchNorm2d"}:
                bz, seq = list(x.size())[:2]
                in_sz = list(x.size())[2:]
                x = x.contiguous().view(bz*seq, *in_sz)
                mean = self.layer(x)
                out_sz = list(mean.size())[1:]
                return mean.contiguous().view(bz, seq, *out_sz), None
            elif m_name == "MultiheadAttention":
                x = x.permute(1, 0, 2) 
                mean, _ = self.layer(x, x, x)
                return mean.permute(1, 0, 2), None
            elif m_name == "LSTM":
                mean, _ = self.layer(x)
            else:
                mean = self.layer(x)
            return mean, None

class ActivationVarPropagationLayer(VarPropagationLayer):
    """
    Specific variance propagation layer for activation functions
    
    Properties:
        inputs: Tensor, input to the activation function from mean propagation stream for computing Jacobian
        exact: bool, (NOT FULLY IMPLEMENTED, only ReLIU) use exact variance propagation through non-linearities
    """
    def __init__(self, layer=None, **kwargs):
        if 'exact' in kwargs:
            self.exact = kwargs['exact']
            del kwargs['exact']
        else:
            self.exact = False
        super(ActivationVarPropagationLayer, self).__init__(layer, **kwargs)
    
    def _call_full_cov(self, x, var):
        if self.exact:
            x, var = self._call_full_cov_exact(x, var)
        else:
            x, var = self._call_full_cov_approx(x, var)
        return x, var
        
    def _call_diag_cov(self, x, var):
        if self.exact:
            x, var = self._call_diag_cov_exact(x, var)
        else:
            x, var = self._call_diag_cov_approx(x, var)
        return x, var
        
    def _call_diag_cov_approx(self, x, var):
        raise NotImplementedError
