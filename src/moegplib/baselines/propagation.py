import time
import torch
import numpy as np
import sys
import os

from moegplib.baselines.layers.activation import LeakyReLUActivationVarPropagationLayer, \
                                        ReLUActivationVarPropagationLayer,\
                                        TanhActivationVarPropagationLayer,\
                                        SigmoidActivationVarPropagationLayer
from moegplib.baselines.layers.fc import DenseVarPropagationLayer, DenseVarPropagationLayer2
from moegplib.baselines.layers.cov import Conv2DVarPropagationLayer
from moegplib.baselines.layers.dropout import DropoutVarPropagationLayer
from moegplib.baselines.layers.flatten import Flatten
from moegplib.baselines.layers.batchnorm import BatchNorm2DVarPropagationLayer
from moegplib.baselines.layers.mha import MHAVarPropagationLayer
from moegplib.baselines.layers.lstm import LSTMVarPropagationLayer

var_layers_dict = {
    'Flatten': Flatten,
    'Dropout': DropoutVarPropagationLayer,
    'ReLU': ReLUActivationVarPropagationLayer,
    'Tanh': TanhActivationVarPropagationLayer,
    'Sigmoid': SigmoidActivationVarPropagationLayer,
    'LeakyReLU': LeakyReLUActivationVarPropagationLayer,
    'Linear': DenseVarPropagationLayer, # 2
    'Conv2d': Conv2DVarPropagationLayer, # 2
    'BatchNorm2d': BatchNorm2DVarPropagationLayer,
    'LSTM': LSTMVarPropagationLayer,
    'MultiheadAttention': MHAVarPropagationLayer
}


# because original Sequential can't handle multiple inputs
class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class VarPropaNet(torch.nn.Module): 
    def __init__(self, model, use_cov, cuda):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VarPropaNet, self).__init__()
        
        # get all layers in the model
        noise_layer_counter = 0
        layer_list = []
        for m in model.modules():
            layer_name = m.__class__.__name__
            if layer_name in ["Sequential", "MHA", "DEEPVO", "_LinearWithBias"]:
                continue

            print("### Re-building layer: {}...".format(layer_name))
            if layer_name == "Dropout":
                # print("dorpout rate: ", m.p)
                init_noise = (noise_layer_counter == 0)
                noise_layer_counter += 1
                layer_list.append(var_layers_dict[layer_name](m, 
                                                            initial_noise=init_noise, 
                                                            use_cov=use_cov, 
                                                            cuda=cuda))
            else:
                layer_list.append(var_layers_dict[layer_name](m, 
                                                            use_cov=use_cov, 
                                                            cuda=cuda))
        self.net = mySequential(*layer_list)
        
    def forward(self, x, var):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.net(x, var)


class UncertaintyPropagator_VIOSOFT(torch.nn.Module):
    """
    class for approximated variance propagation for visual odometry network VIOSOFT 
    
    Inputs:
        - model: a loaded pre-trained VIOSOFT network
        - use_cov: whether to use full cov or diagonal for propagation 
    """
    def __init__(self,
                 model,
                 use_cov: bool=False,
                 cuda: bool=True,
                 debug: bool=False,
                 **kwargs):
        super(UncertaintyPropagator_VIOSOFT, self).__init__()
        self.cuda = cuda
        self.use_cov = use_cov
        self.debug = debug
       
        # build variance propagator for img_encoder
        self.img_encoder_namelist = ["conv1", "conv2", "conv3", "conv3_1", "conv4", "conv4_1", "conv5", "conv5_1", "conv6"]
        self.img_encoder_list = []
        for idx, conv_name in enumerate(self.img_encoder_namelist):
            init_noise = (idx == 0)
            for layer in model._modules[conv_name]:
                l_name = layer.__class__.__name__
                if self.debug:
                    print("### Re-building layer: {} in {} of img_encoder...".format(l_name, conv_name))
                if l_name == "Dropout":
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, initial_noise=init_noise, use_cov=use_cov, cuda=cuda))
                else:
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda))

        self.img_encoder_var = mySequential(*self.img_encoder_list)
        self.rnnIMU = model._modules["rnnIMU"]
        self.linear_imu = model._modules["linear_imu"]
        layer = model._modules["linear_cnn"]
        l_name = layer.__class__.__name__
        self.linear_cnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        layer = model._modules['linear_soft']
        l_name = layer.__class__.__name__
        self.linear_soft_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        layer = model._modules["sigmoid"]
        l_name = layer.__class__.__name__
        self.sigmoid_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        layer = model._modules["rnn"]
        l_name = layer.__class__.__name__
        self.rnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        layer = model._modules["rnn_drop_out"]
        l_name = layer.__class__.__name__
        self.rnn_drop_out_var = var_layers_dict[l_name](layer, initial_noise=False, use_cov=use_cov, cuda=cuda)
        layer = model._modules["linear"]
        l_name =layer.__class__.__name__
        self.linear_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)

    def forward(self, x, imu):
        # stack image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2) 
        batch_size = x.size(0)
        seq_len = x.size(1)

        with torch.no_grad():
            # cnn
            var = None
            mean_img, var = self.img_encoder_var(x, var) 
            mean_img = mean_img.view(batch_size, seq_len, -1)
            var = var.view(batch_size, seq_len, -1)
            mean_img, var = self.linear_cnn_var(mean_img, var)

            # imu
            imu_out, _ = self.rnnIMU(imu)
            imu_out = imu_out.contiguous().view(batch_size, seq_len, -1)
            mean_imu = self.linear_imu(imu_out)

        # linear_soft and sigmoid
        mean_concat = torch.cat((mean_img, mean_imu), 2)
        var_concat = torch.cat((var, torch.zeros_like(mean_imu, requires_grad=True)), 2)
        mean_soft, var_soft = self.linear_soft_var(mean_concat, var_concat)
        mean, var = self.sigmoid_var(mean_soft, var_soft)
        
        # final lstm
        mean, var = self.rnn_var(mean, var)
        mean, var = self.rnn_drop_out_var(mean, var)
        with torch.no_grad():
            mean, var = self.linear_var(mean, var)
        return mean, var


class UncertaintyPropagator_VINET(torch.nn.Module):
    
    """
    class for approximated variance propagation for visual odometry network VINET 
    
    Inputs:
        - model: a loaded pre-trained VINET network
        - use_cov: whether to use full cov or diagonal for propagation 
    """
    def __init__(self,
                 model,
                 use_cov: bool=False,
                 cuda: bool=True,
                 debug: bool=False,
                 **kwargs):
        super(UncertaintyPropagator_VINET, self).__init__()
        self.cuda = cuda
        # self.model = model
        self.use_cov = use_cov
        self.debug = debug
       
        # build variance propagator for img_encoder
        self.img_encoder_namelist = ["conv1", "conv2", "conv3", "conv3_1", "conv4", "conv4_1", "conv5", "conv5_1", "conv6"]
        self.img_encoder_list = []
        for idx, conv_name in enumerate(self.img_encoder_namelist):
            init_noise = (idx == 0)
            for layer in model._modules[conv_name]:
                l_name = layer.__class__.__name__
                if self.debug:
                    print("### Re-building layer: {} in {} of img_encoder...".format(l_name, conv_name))
                if l_name == "Dropout":
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, initial_noise=init_noise, use_cov=use_cov, cuda=cuda))
                else:
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda))

        self.img_encoder_var = mySequential(*self.img_encoder_list)

        # IMU
        self.rnnIMU = model._modules["rnnIMU"]
        self.linear_imu = model._modules["linear_imu"]

        # build var layer for linear_cnn
        layer = model._modules["linear_cnn"]
        l_name = layer.__class__.__name__
        self.linear_cnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)

        # build core LSTM var layer
        layer = model._modules["rnn"]
        l_name = layer.__class__.__name__
        self.rnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        
        # build rnn_drop_out var layer
        layer = model._modules["rnn_drop_out"]
        l_name = layer.__class__.__name__
        self.rnn_drop_out_var = var_layers_dict[l_name](layer, initial_noise=False, use_cov=use_cov, cuda=cuda)

        # build final linear var layer
        layer = model._modules["linear"]
        l_name =layer.__class__.__name__
        self.linear_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)

    def forward(self, x, imu):
        # stack image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2) 
        batch_size = x.size(0)
        seq_len = x.size(1)
        with torch.no_grad():
            # cnn
            var = None
            mean_img, var = self.img_encoder_var(x, var) 
            mean_img = mean_img.view(batch_size, seq_len, -1)
            var = var.view(batch_size, seq_len, -1)
            mean_img, var = self.linear_cnn_var(mean_img, var)

            # imu
            imu_out, _ = self.rnnIMU(imu)
            imu_out = imu_out.contiguous().view(batch_size, seq_len, -1)
            mean_imu = self.linear_imu(imu_out)
        mean_concat = torch.cat((mean_img, mean_imu), 2)
        var_concat = torch.cat((var, torch.zeros_like(mean_imu, requires_grad=True)), 2)

        # final lstm
        mean, var = self.rnn_var(mean_concat, var_concat)
        mean, var = self.rnn_drop_out_var(mean, var)
        with torch.no_grad():
            mean, var = self.linear_var(mean, var)
        return mean, var
    

class UncertaintyPropagator_MHA(torch.nn.Module):
    """
    class for approximated variance propagation for visual odometry network MHA 
    
    Inputs:
        - model: a loaded pre-trained MHA network
        - use_cov: whether to use full cov or diagonal for propagation 
    """
    def __init__(self,
                 model,
                 use_cov: bool=False,
                 cuda: bool=True,
                 debug: bool=False,
                 **kwargs):
        super(UncertaintyPropagator_MHA, self).__init__()
        self.cuda = cuda
        # self.model = model
        self.use_cov = use_cov
        self.debug = debug
       
        # build variance propagator for img_encoder
        self.img_encoder_namelist = ["conv1", "conv2", "conv3", "conv3_1", "conv4", "conv4_1", "conv5", "conv5_1", "conv6"]
        self.img_encoder_list = []
        for idx, conv_name in enumerate(self.img_encoder_namelist):
            init_noise = (idx == 0)
            for layer in model._modules[conv_name]:
                l_name = layer.__class__.__name__
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, initial_noise=init_noise, use_cov=use_cov, cuda=cuda))
                else:
                    self.img_encoder_list.append(var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda))

        self.img_encoder_var = mySequential(*self.img_encoder_list)

        # IMU
        self.rnnIMU = model._modules["rnnIMU"]
        self.linear_imu = model._modules["linear_imu"]

        # build var layer for linear_cnn
        layer = model._modules["linear_cnn"]
        l_name = layer.__class__.__name__
        self.linear_cnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)

        # build MultiheadAttention var layer
        layer = model._modules["multihead_attn"]
        # layer.train()
        l_name = layer.__class__.__name__
        self.multihead_attn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda) 

        # build core LSTM var layer
        layer = model._modules["rnn"]
        l_name = layer.__class__.__name__
        self.rnn_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)
        
        # build rnn_drop_out var layer
        layer = model._modules["rnn_drop_out"]
        l_name = layer.__class__.__name__
        self.rnn_drop_out_var = var_layers_dict[l_name](layer, initial_noise=False, use_cov=use_cov, cuda=cuda)

        # build final linear var layer
        layer = model._modules["linear"]
        l_name =layer.__class__.__name__
        self.linear_var = var_layers_dict[l_name](layer, use_cov=use_cov, cuda=cuda)

    def forward(self, x, imu):
        # stack image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2) 
        batch_size = x.size(0)
        seq_len = x.size(1)

        with torch.no_grad():
            # cnn
            var = None
            mean_img, var = self.img_encoder_var(x, var) 
            mean_img = mean_img.view(batch_size, seq_len, -1)
            var = var.view(batch_size, seq_len, -1)
            mean_img, var = self.linear_cnn_var(mean_img, var)

            # imu
            imu_out, _ = self.rnnIMU(imu)
            imu_out = imu_out.contiguous().view(batch_size, seq_len, -1)
            mean_imu = self.linear_imu(imu_out)

        # MHA
        mean_concat = torch.cat((mean_img, mean_imu), 2)
        var_concat = torch.cat((var, torch.zeros_like(mean_imu, requires_grad=True)), 2)
        mean, var = self.multihead_attn_var(mean_concat, var_concat)

        # final lstm
        mean, var = self.rnn_var(mean, var)
        mean, var = self.rnn_drop_out_var(mean, var)
        with torch.no_grad():
            mean, var = self.linear_var(mean, var)
        return mean, var


class UncertaintyPropagator():
    """
    class for approximated variance propagation
    
    Inputs:
        - model: torch.nn.Sequential() (or later include torch.nn.Module and ModuleList)
        - mc_samples: number of samples for Monte Carlo dropout
        - use_cov: whether to use full cov or diagonal for propagation 
    """
    def __init__(self,
                 model,
                 mc_samples: int=10,
                 use_cov: bool=False,
                 cuda: bool=True,
                 **kwargs):

        self.cuda = cuda
        self.model = model
        self.mc_samples = mc_samples
        self.use_cov = use_cov
        
        # build model
        self.unc_model = VarPropaNet(self.model, self.use_cov, self.cuda)
        if self.cuda:
            self.unc_model.double().cuda()
        
    def predict_mcd(self, X, MCD=10, return_runtime=True):
         # predict with mc sampling model
        start = time.time()
        result_tmp = [self.model(X) for _ in range(MCD)]
        rt = time.time() - start

        # compute std and mean
        result = []
        for r in result_tmp:
            result.append(r.cpu().data.numpy())
        means = np.mean(result)
        result_var = np.var(result)
        if return_runtime:
            return means, result_var, rt
        else: 
            return means, result_var

    def predict_avp(self, X, return_runtime=True):
        start = time.time()
        means, result_var = self.unc_model(X, None)
        rt = time.time() - start
        if return_runtime:
            return means, result_var, rt
        else: 
            return means, result_var
