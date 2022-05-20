import time
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from layer_utils import get_jacobian_loop

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Linear(4, 8),
                        nn.Linear(8, 512)
                        # nn.Linear(16, 2)
                        )
        self.multihead_attn = nn.MultiheadAttention(512, 8)
        self.rnn = nn.LSTM(input_size=512,  
                            hidden_size=1000, 
                            num_layers=2,
                            dropout=0,
                            batch_first=True,
                            bidirectional=True)
        self.layers_init()

    def layers_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

    def forward(self, x):
        self.l_input = []
        self.l_output = []
        self.l_input.append(x.requires_grad_(True))
        lin = self.layer1(x)
        self.l_output.append(lin.requires_grad_(True))
        
        self.l_input.append(lin)
        mha, _ = self.multihead_attn(lin, lin, lin)
        self.l_output.append(mha.requires_grad_(True))
        
        self.l_input.append(mha)
        lstm, _ = self.rnn(mha)
        self.l_output.append(lstm.requires_grad_(True))
        return lstm

    def backward(self, inp):
        for m in self.modules():
            if m.__class__.__name__ in {"Linear", "MultiheadAttention", "LSTM"}:
                print("### module name: ", m.__class__.__name__)
                if m.__class__.__name__ == "MultiheadAttention":
                    nout = m.embed_dim 
                    out, _ = m(inp, inp, inp)
                elif m.__class__.__name__ == "LSTM":
                    nout = m.hidden_size * (2 if m.bidirectional else 1) 
                    out, _ = m(inp)
                else:
                    nout = m.out_features 
                    out = m(inp)
                print("Output dim: {}".format(nout))
                jcb_bz = 1
                start = time.time()
                J =  get_jacobian_loop(m, inp, nout, jcb_bz=jcb_bz, return_out=False)
                jcb_time = time.time() - start
                print("jcb_batch_size: {}, input size: {}, output size: {}".format(jcb_bz, inp.size(), out.size()))
                print("Jacobian (time: {:.3f}s) of out wrt. inp is:".format(jcb_time))
                print(J.size())
                inp = out