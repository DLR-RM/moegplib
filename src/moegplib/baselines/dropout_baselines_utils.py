import numpy as np 
import torch
import time
import sys
import os

from moegplib.baselines.propagation import UncertaintyPropagator_MHA, UncertaintyPropagator_VINET, UncertaintyPropagator_VIOSOFT


class dropout_baseline_prediction(object):
    def __init__(self, model, model_name):
        super(dropout_baseline_prediction, self).__init__()
        self.model = model
        self.model_name = model_name
        
        self.model.eval() # in eval() mode by default

    def standard_pred(self, model_input):
        if self.model_name == "DEEPVO":
            im = model_input
            batch_predict_pose = self.model.forward(im)
        else:
            im, x = model_input
            batch_predict_pose = self.model.forward(im, x)  
        return batch_predict_pose
    
    def mc_dropout_pred(self, model_input, num_samples=30):        
        # to turn off batch norm and leave only dropout on
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        for i in range(num_samples):
            batch_predict_pose = self.standard_pred(model_input)

            if i == 0:
                pred_sum = batch_predict_pose.detach().clone().unsqueeze(0)
            else:
                pred_sum = torch.cat((pred_sum, batch_predict_pose.detach().clone().unsqueeze(0)))
        avg_pred_pose = pred_sum 
        var_pred_pose = torch.var(pred_sum, dim=0)
        return avg_pred_pose, var_pred_pose

    def var_propagation_pred(self, model_input):
       
        if self.model_name == 'MHA':
            self.var_propagate_net = UncertaintyPropagator_MHA(self.model)
        elif self.model_name == 'VIOSOFT':
            self.var_propagate_net = UncertaintyPropagator_VIOSOFT(self.model)
        elif self.model_name == 'VINET':
            self.var_propagate_net = UncertaintyPropagator_VINET(self.model)
        self.var_propagate_net.eval()
        # with torch.no_grad():
        if self.model_name == "DEEPVO":
            im = model_input
            pred_pose, pred_pose_var = self.var_propagate_net(im)
        else:
            im, imu = model_input
            pred_pose, pred_pose_var = self.var_propagate_net(im, imu)  

        return pred_pose, pred_pose_var