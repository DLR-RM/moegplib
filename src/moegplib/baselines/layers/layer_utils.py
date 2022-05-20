import torch

def get_jacobian_loop(module, x, nout, jcb_bz=1, return_out=False):
    """function to compute Jacobian of output of one module wrt. the input in a 
    mini-batch manner based on torch autograd, reference: 
    1. https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#jacobian

    :param module (nn.module): one nn.module in pytorch
    :param x (tensor [batch_size, seq_len, *input_feat_dim]): input
    :param nout (int): feature dimension of output of module
    :parm jcb_bz (int): size of mini-batch in computing Jacobian, larger number consume more memory
    :param return_out (bool): to return the output of module along with Jacobian or not, if True more memory consumption

    :return Jacobian matrix (tensor [batch_size, seq_len, nout, *input_feat_dim])
    if return_out is True, return (out, Jacobian)
    """
    bz, sq = x.shape[:2] # [bz, seq]
    assert bz % jcb_bz == 0, "batch-size of jcb must be divisible by batch-size of data!"
    for jcb_start_idx in range(0, bz, jcb_bz):
        x_batch = x[jcb_start_idx: jcb_start_idx + jcb_bz, :].detach() 
        # print("mini jcb batch input size: ", x_batch.size())
        if jcb_start_idx == 0:
            if return_out:
                out, jcb = get_jacobian_tile(module, x_batch, nout) # [jcb_bz, seq, nout, *in_sz]
            else:
                jcb = get_jacobian_tile(module, x_batch, nout, return_out=False) # [jcb_bz, seq, nout, *in_sz]

        else:
            if return_out:
                out_tmp, jcb_tmp = get_jacobian_tile(module, x_batch, nout)
                jcb = torch.cat((jcb, jcb_tmp), dim=0)
                out = torch.cat((out, out_tmp), dim=0)
            else:
                jcb_tmp = get_jacobian_tile(module, x_batch, nout, return_out=False)
                jcb = torch.cat((jcb, jcb_tmp), dim=0)
    if return_out: 
        return out, jcb 
    else:
        return jcb


def get_jacobian_batch_loop(module, x, nout, feat_start_idx=2, return_out=True):
    """function to compute Jacobian of output of one module wrt. the input in 
    single-batch manner by tiling the input according to the output dim, reference: 
    1. https://gist.github.com/MasanoriYamada/d1d8ca884d200e73cca66a4387c7470a

    :param module (nn.module): one nn.module in pytorch
    :param x (tensor [1, seq_len, *input_feat_dim]): input
    :param nout (int): feature dimension of output of module
    :parm feat_start_idx (int): the position in size of input where input_feat starts
    :param return_out (bool): to return the output of module along with Jacobian or not, if True more memory consumption

    :return Jacobian matrix (tensor [batch_size, seq_len, nout, *input_feat_dim])
    if return_out is True, return (out, Jacobian)
    """
    pass
    


def get_jacobian_tile(module, x, nout, feat_start_idx=2, return_out=True):
    """function to compute Jacobian of output of one module wrt. the input in 
    single-batch manner by tiling the input according to the output dim, reference: 
    1. https://gist.github.com/MasanoriYamada/d1d8ca884d200e73cca66a4387c7470a

    :param module (nn.module): one nn.module in pytorch
    :param x (tensor [batch_size, seq_len, *input_feat_dim]): input
    :param nout (int): feature dimension of output of module
    :parm feat_start_idx (int): the position in size of input where input_feat starts
    :param return_out (bool): to return the output of module along with Jacobian or not, if True more memory consumption

    :return Jacobian matrix (tensor [batch_size, seq_len, nout, *input_feat_dim])
    if return_out is True, return (out, Jacobian)
    """
    module.train()
    module.zero_grad()
    b_dim = feat_start_idx
    batch_seq = x.shape[:b_dim]
    x_shape = x.shape[b_dim:]
    x = x.unsqueeze(b_dim)
    x = x.repeat(1, 1, nout, *(1,)*len(x.shape[b_dim+1:])) 
    if module.__class__.__name__ == "MultiheadAttention":
        x = x.permute(1, 0, 2, 3)
        x = x.contiguous().view(batch_seq[1], batch_seq[0]*nout, *x_shape) 
        x.requires_grad_(True)
        x.retain_grad()
        y, _ = module(x, x, x) 
        y = y.contiguous().view(batch_seq[1], batch_seq[0], nout, nout)
        y = y.permute(1, 0, 2, 3) 
    elif module.__class__.__name__ == "LSTM":
        x = x.permute(0, 2, 1, 3)
        x = x.view(batch_seq[0]*nout, batch_seq[1],  *x_shape) 
        x.requires_grad_(True)
        x.retain_grad()
        y, _ = module(x) 
        y = y.contiguous().view(batch_seq[0], nout, batch_seq[1], nout)
        y = y.permute(0, 2, 1, 3) 
    else:
        x.requires_grad_(True)
        x.retain_grad()
        y = module(x)
    y_eye = torch.eye(nout).view(1, 1, nout, nout).repeat(*batch_seq, 1, 1).cuda() 
    y.backward(y_eye, retain_graph=True)
    if return_out:
        return y[:, :, 0, :].squeeze(2), x.grad.view(*batch_seq, nout, *x_shape)
    else:
        if module.__class__.__name__ == "MultiheadAttention":
            jcb = x.grad.view(batch_seq[1], batch_seq[0], nout, *x_shape)
            return jcb.permute(1, 0, 2, 3)
        elif module.__class__.__name__ == "LSTM":
            jcb = x.grad.view(batch_seq[0], nout, batch_seq[1], *x_shape)
            return jcb.permute(0, 2, 1, 3)
        else:
            return x.grad
