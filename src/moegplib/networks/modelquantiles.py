""" model quanitles: quantities that are needed by NTK GP.
"""
import numpy as np
import torch
import time
import gc
import torch.nn.functional as F

from torch.distributions import Normal, Bernoulli
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from scipy.special import logsumexp
from torch.utils.data import DataLoader

from moegplib.utils.logger import SaveAndLoad, BoardLogger
from moegplib.networks.detr.boxops_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

torch.set_default_dtype(torch.double)


class QuantilesBase:
    """A parent class to model quantiles.     
    """
    def __init__(self, model, data, delta=1, sigma_noise=1, seed=77, 
                 is_mjacob=False, targetout=0, devices='cpu'):
        """Initialization

        Args:
            model ([type]): [pytorch model class.]
            data ([tuple]): [(xdata, ydata)]
            delta ([type]): [prior precision]
            sigma_noise (int, optional): [noise parameter]. Defaults to 1.
            seed (int, optional): [seed]. Defaults to 77.
            targetout (int, optional): [considered output number]. Defaults to 0.
            outputdim (int, optional): [totol output dimension]. Defaults to 1.
            is_mjacob (bool): multi-dimensional jacobians?
            devices (str, optional): [gpu device]. Defaults to 'cpu'.
        """
        torch.manual_seed(seed)
        self.model = model
        self.data = data
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.seed = seed
        self.is_mjacob = is_mjacob
        self.targetout = targetout
        self.device = devices
        
    def _compute_m0(self):
        """Computes the prior mean of GP.

        Returns:
            m_0 (torch.Tensor): Prior mean of GP.
        """
        raise NotImplementedError()
    
    def _compute_S0(self):
        """Computes the hyperparameter delta (the variance  of the kernel).

        Returns:
            S_0 (torch.Tensor): the variance  of the kernel.
        """
        raise NotImplementedError()
    
    def _compute_he(self):
        """Computes HE - the K x K noise precision matrix.
        HE is the hessian of the loss w.r.t function or ourput.
        It can be treated as a simple scalar/7x1 vector for easiness (assuming diag).

        Returns:
            HE (torch.Tensor): the K x K noise precision matrix in diag format
        """
        raise NotImplementedError()
    
    def _compute_snoise(self):
        """Computes s_noise - the K x K noise matrix and 1/sqrt(he).
        
        It ends up being K x 1 vector with a uniform assumption.

        Returns:
            s_noise (torch.Tensor): the noise sigma.
        """
        raise NotImplementedError()
    
    def _compute_de(self):
        """Computes DE - the residual vector K x 1.
        
        It stacks the batched assumption and puts into K Batch x 1 vector.

        Returns:
            DE (torch.Tensor): the residual vector.
        """
        raise NotImplementedError()
    
    def _compute_dftheta(self, Xtest=None):
        """Computes dftheta/jacobian - the derivative of output by weight per single data point.

        Args:
            Xtest (torch.Tensor, optional): Input data for testing. Defaults to None.

        Returns:
            dftheta (torch.Tensor): the jacobian matrix.
        """
        raise NotImplementedError()
        
    def _compute_ytransform(self, Xhat, minux):
        """Computes the ytransfrom using the jacobians.

        Args:
            Xhat (torch.Tensor): dftheta/Jacobian matrix.
            minux (torch.Tensor): the error between predictive and ground truth.
            
        Returns:
            yhat (torch.Tensor): the transformed y transformed.
        """
        raise NotImplementedError()
    
    def projection(self, Xtest=None, is_lightweight=True):
        """Computes the linear projection quantities. 
        
        Args:
            Xtest (torch.Tensor, optional): Test data to transform. Defaults to None.
            is_lightweight (bool): Decides which to return.

        Returns:
            (tuple): A tuple (Xhat, yhat, snoise, m0, s0).
                     Xhat: Jacobian matrix.
                     yhat: transformed output.
                     snoise: data uncertainty.
                     m0: GP prior mean.
                     S0: GP hyperparameter.
        """
        raise NotImplementedError()
    
    
class DetectronQuantiles(QuantilesBase):
    """ Detectron model quantiles class
    """
    def __init__(self, model, data, classout=0, regout=0, 
                 task="regression", is_mjacob=False, 
                 devices='cuda'):
        """[summary]
        Data should be a batch data from Detectron data_loader.
        Link: https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
        """
        super().__init__(model, data, seed=77, is_mjacob=is_mjacob, devices=devices)
        self.task = task
        self.classout = classout
        self.regout = classout
        #self.mapweight = self._compute_theta_map()
        #self.p = len(self.mapweight)

    @torch.no_grad() 
    def _compute_de(self):
        """Computes the residual between output and ground truth
        """
        # initialization
        self.model.eval()

        # model prediction
        images = self.model.preprocess_image(self.data)
        output = self.model.detr(images)
        scores_batch, labels_batch = F.softmax(output["pred_logits"], dim=-1)[:, :, :-1].max(-1)
        targets = self._prepare_targets()

        # choosing the task (bounding box regression or object classification)
        if self.task == "regression":
            residual = list()
            #for groundtruth, pred_logits, pred_boxes in zip(targets, output["pred_logits"], output["pred_boxes"]):
            for groundtruth, scores, labels, pred_boxes in zip(targets, scores_batch, labels_batch, output["pred_boxes"]):
                # processing of the raw network output
                classindex = self._detr_prediction_parser(class_score=scores, threshold=0.5)

                # checking the label
                if classindex is not None:
                    objectlabels = labels.squeeze()[classindex]
                    do_jacobian = objectlabels==self.classout
                    do_jacobian = do_jacobian.squeeze().cpu().numpy()

                    # computing the residuals
                    if do_jacobian:
                        predbox = pred_boxes[classindex].squeeze().to(self.device)
                        ytbox = groundtruth['boxes'].squeeze().to(self.device)
                        residual.append(predbox.unsqueeze(0) - ytbox.unsqueeze(0))
        elif self.task == "classification":
            # NOTE: classification does not require implementation of residuals
            # since we take the temperature scaling approach with scale factor
            # as function of variance on the GP regression of logits (classification as regression).
            return None
        else:
            raise AttributeError

        # checking for the exceptions that the current batch doesnt contain our data
        if residual:
            return torch.cat(residual).to(self.device) # X x 4 matrix
        else:
            return None

    def _compute_theta_map(self):
        """Computes theta map - estimates of network weights with MAP principle.
        P x 1 in format, for P being the total number of parameters.

        Returns:
            theta_map (torch.Tensor): the map estimates of parameters
        """
        theta_map = list()
        for p in self.model.parameters():
            if p.grad is not None:
                theta_map.append(p.flatten())
        theta_map = torch.cat(theta_map)
        #theta_map = torch.cat([p.flatten() for p in self.model.parameters()])
        return theta_map.to(self.device)

    def _detr_prediction_parser(self, class_score, threshold=0.5):
        """[summary]

        Args:
            prediction (instance): "instance" of detectron 2
            threshold (float, optional): [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """        
        # only for class score higher
        logic = class_score > threshold
        classind = torch.nonzero(logic.squeeze())
        if classind.nelement():
            return classind
        else:
            return None

    def _prepare_targets(self):
        targets = [x["instances"] for x in self.data]
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float) #, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def _compute_dftheta(self, Xtest=None):
        """
        NOTE: we implement is_mjacob later
        """
        # initialization (self.task = task or not)
        data = self.data # train data
        if Xtest is not None:
            data = Xtest # output data
        jacobO = list() 
        jacobL = None
        idx, select_idx = 0, list()

        # iterate per data
        for xi in data:
            # model output
            self.model.zero_grad()
            images = self.model.preprocess_image([xi])
            prediction = self.model.detr(images)
            scores, labels = F.softmax(prediction["pred_logits"], dim=-1)[:, :, :-1].max(-1)
            
            # processing of the raw network output
            classindex = self._detr_prediction_parser(class_score=scores, threshold=0.7)
            if classindex is not None:
                objectlabels = labels.squeeze()[classindex]
                do_jacobian = objectlabels==self.classout
                do_jacobian = do_jacobian.squeeze().cpu().numpy()
                print(do_jacobian, objectlabels, "####### sources of error #########")

                # separate classification and regression
                if self.task == "regression":
                    if do_jacobian:
                        predbox = prediction["pred_boxes"][0, classindex] # 1 x 10 x 4
                        temp = torch.zeros_like(predbox) # select grad_tensors
                        temp[:, :, self.regout] = 1
                        predbox.backward(temp, retain_graph=True)
                        jacobL = self._jacobian()
                        select_idx.append(idx)
                    else:
                        print("regression-NBNNNN")
                        jacobL = None
                elif self.task == "classification":
                    if do_jacobian:
                        temp = torch.zeros_like(scores) # select grad_tensors
                        temp[0, classindex]= 1 #NOTE: this is correct?
                        scores.backward(temp, retain_graph=True)
                        jacobL = self._jacobian()
                        select_idx.append(idx)
                    else:
                        print("classification-NBNNNN")
                        jacobL = None
                else:
                    raise AttributeError

            # storing the jacobians
            if jacobL is not None:
                jacobO.append(jacobL)

            # update index
            idx = idx + 1
            
        if select_idx:
            jacobO = torch.stack(jacobO, dim=1).T.unsqueeze(0)
            return jacobO.to(self.device)
        else:
            return None

    def _jacobian(self):
        """ Implementation of Jacobian computations
        We check for an exception that gradient is None.
        """
        jacobL = list()
        for p in self.model.parameters():
            if p.grad is not None:
                jacobL.append(p.grad.data.flatten())
        jacobL = torch.cat(jacobL)
        return jacobL
    
    @torch.no_grad() 
    def _compute_ytransform(self, Xhat, minux):
        if self.task == "regression":
            yhat = torch.zeros(1, Xhat.shape[1]).to(self.device)
            yhat[0, :] = Xhat[0, :, :] @ self.mapweight - minux[:, self.regout]
        elif self.task == "classification":
            if minux is not None:
                raise AttributeError
            yhat = torch.zeros(1, Xhat.shape[1]).to(self.device)
            yhat[0, :] = Xhat[0, :, :] @ self.mapweight
        else:
            raise AttributeError
        return yhat

    def projection(self, Xtest=None, is_lightweight=True):
        
        Xhat = self._compute_dftheta(Xtest=Xtest)

        if Xtest is not None:
            return (Xhat, None, None, None, None)
        else:
            if Xhat is not None:
                # computing the map weights
                self.mapweight = self._compute_theta_map()
                self.p = len(self.mapweight)

                # other model quantiles
                minux = self._compute_de()
                
                # yhat: transformed y
                yhat = self._compute_ytransform(Xhat, minux)
                
                # delete unused variables            
                del minux
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()   
                
                # Xhat and Jacobians
                return (Xhat, yhat, None, None, None)
            else:
                return (None, None, None, None, None)


class ModelQuantiles():
    """This class computes required model quantities including Jacobians and MAP estimates.
    All functions are defined per point data. We update this module for gpu compatibility.
    
    Here, we assume an MSE loss function.
    """
    def __init__(self, model, data, delta=1, sigma_noise=1, seed=77, 
                 is_mjacob=False, targetout=0, outputdim=1, devices='cpu'):
        """Initialization

        Args:
            model (torch.nn.module): pytorch model class.
            data (tuple): data tuple (xdata, ydata).
            delta (float, optional): prior precision. Defaults to 1.
            sigma_noise (int, optional): noise parameter. Defaults to 1.
            seed (int, optional): seed. Defaults to 77.
            targetout (int, optional): considered output number. Defaults to 0.
            outputdim (int, optional): total output dimension. Defaults to 1.
            is_mjacob (bool, optional): USe multi-dimensional jacobians?. Defaults to False.
            devices (str, optional): Gpu device either 'cuda' or 'cpu'. Defaults to 'cpu'.
        """
        torch.manual_seed(seed)
        self.is_mjacob = is_mjacob
        self.devices = devices
        self.model = model
        self.xt, self.yt = data
        self.mapweight = self._compute_theta_map()
        self.p = len(self.mapweight)
        self.targetout = targetout
        self.k = outputdim
        if self.yt is not None:  
            self.k = self.yt.shape[1] # number of outputs
            self.batchsize = len(self.yt)
            self.sigma_noise = torch.from_numpy(np.asarray(sigma_noise))
            self.sn = self.sigma_noise ** 2 
            self.bn = 1 / self.sn  
            self.delta = torch.from_numpy(np.asarray(delta))
            self.he = self._compute_he() 
        if self.targetout > self.k:
            raise ValueError("Mis-specification of the output \
                             (network doesnt have the output number!")
        
    def _compute_m0(self):
        """Computes the prior mean of GP.

        Returns:
            m_0 (torch.Tensor): Prior mean of GP.
        """
        m_0 = torch.zeros(self.p)
        return m_0.to(self.devices)
    
    def _compute_S0(self):
        """Computes the hyperparameter delta (the variance  of the kernel).

        Returns:
            S_0 (torch.Tensor): the variance  of the kernel.
        """
        S_0 = 1 / self.delta * torch.ones(self.p)
        return S_0.to(self.devices)
    
    def _compute_he(self):
        """Computes HE - the K x K noise precision matrix.
        HE is the hessian of the loss w.r.t function or ourput.
        It can be treated as a simple scalar/7x1 vector for easiness (assuming diag).

        Returns:
            HE (torch.Tensor): the K x K noise precision matrix in diag format
        """
        he = torch.ones(self.k) * self.bn  # a strong assumption that the loss is MSE
        return he.to(self.devices)
    
    def _compute_snoise(self):
        """Computes s_noise - the K x K noise matrix and 1/sqrt(he).
        It ends up being K x 1 vector with a uniform assumption.

        Returns:
            s_noise (torch.Tensor): the noise sigma.
        """
        s_noise = 1 / torch.sqrt(self.he)
        return s_noise.to(self.devices)
    
    def _compute_theta_map(self):
        """Computes theta map - estimates of network weights with MAP principle.
        P x 1 in format, for P being the total number of parameters.

        Returns:
            theta_map (torch.Tensor): the map estimates of parameters
        """
        theta_map = torch.cat([p.flatten() for p in self.model.parameters()])
        return theta_map
    
    def _compute_de(self):
        """Computes DE - the residual vector K x 1. We assume MSE error. 
        It stacks the batched assumption and puts into K Batch x 1 vector.

        Returns:
            DE (torch.Tensor): the residual vector.
        """
        # compute output from a model and dataset
        self.model.eval()
        pred = self.model(self.xt)
        
        # compute the residual given the loss
        assert(pred.shape != self.yt.shape, "Shape mismatch between pred and output")
        de = self.bn * (pred - self.yt)
        de = torch.reshape(de, (self.batchsize, self.k))
        return de.to(self.devices)
    
    def _compute_dftheta(self, Xtest=None):
        """Computes dftheta/jacobian - the derivative of output by weight per single data point.

        Args:
            Xtest (torch.Tensor, optional): Input data for testing. Defaults to None.

        Returns:
            dftheta (torch.Tensor): the jacobian matrix.
        """
        # initialization
        xstar = self.xt
        if Xtest is not None:
            xstar = Xtest
            
        # multiple output-setup: iteratute through all dim.
        if self.is_mjacob:
            dftheta = torch.zeros(self.k, len(xstar), 
                                  self.p).to(self.devices)
            for i in range(self.k):
                jacobO = torch.zeros(len(xstar), self.p).to(self.devices)
                idx = 0
                for xi in xstar:
                    self.model.zero_grad()
                    output = self.model.forward(xi.double())
                    temp = torch.zeros_like(output)
                    temp[i] = 1 # selecting the variable
                    output.backward(temp, retain_graph=True)
                    jacobL = torch.cat([p.grad.data.flatten() \
                        for p in self.model.parameters()])
                    jacobO[idx, :] = jacobL
                    idx = idx + 1 
                dftheta[i, :, :] = jacobO
        
        # selecting a specific output dimension.
        else:
            dftheta = torch.zeros(1, len(xstar), 
                                  self.p).to(self.devices)
            jacobO = torch.zeros(len(xstar), self.p).to(self.devices)
            idx = 0
            for xi in xstar:
                self.model.zero_grad()
                output = self.model.forward(xi.double())
                temp = torch.zeros_like(output)
                temp[self.targetout] = 1 # selecting the variable
                output.backward(temp, retain_graph=True)
                jacobL = torch.cat([p.grad.data.flatten() \
                    for p in self.model.parameters()])
                jacobO[idx, :] = jacobL
                idx = idx + 1
            dftheta[0, :, :] = jacobO
        return dftheta
        
    def _compute_ytransform(self, Xhat, minux):
        """Computes the ytransfrom using the jacobians.

        Args:
            Xhat (torch.Tensor): dftheta/Jacobian matrix.
            minux (torch.Tensor): the error between predictive and ground truth.
            
        Returns:
            yhat (torch.Tensor): the transformed y transformed.
        """
        # multiple output setup.
        if self.is_mjacob:
            yhat = torch.zeros(self.k, Xhat.shape[1]).to(self.devices)
            for p in range(self.k):
                yhat[p, :] = Xhat[p, :, :] @ self.mapweight \
                    - minux[:, p]
        # specified output setup.
        else:
            yhat = torch.zeros(1, Xhat.shape[1]).to(self.devices)
            yhat[0, :] = Xhat[0, :, :] @ self.mapweight \
                    - minux[:, self.targetout]
        return yhat
    
    def projection(self, Xtest=None, is_lightweight=True):
        """Computes the linear projection quantities. 
        
        Args:
            Xtest (torch.Tensor, optional): Test data to transform. Defaults to None.
            is_lightweight (bool): Decides which to return.

        Returns:
            (tuple): A tuple (Xhat, yhat, snoise, m0, s0).
                     Xhat: Jacobian matrix.
                     yhat: transformed output.
                     snoise: data uncertainty.
                     m0: GP prior mean.
                     S0: GP hyperparameter.
        """        
        # Xhat: Jacobains (J(x))
        Xhat = self._compute_dftheta(Xtest=Xtest)
        if Xtest is not None:
            return (Xhat, None, None, None, None)
        
        # other model quantiles
        de = self._compute_de()
        he = self._compute_he()
        minux = de * 1/he
        
        # yhat: transformed y
        yhat = self._compute_ytransform(Xhat, minux)
        
        # delete unused variables            
        del de, he, minux
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()   
        
        # Xhat nad Jacobians
        if is_lightweight:
            return (Xhat, yhat, None, None, None)
        
        # other model quantiles    
        m_0 = self._compute_m0()
        S_0 = self._compute_S0()
        s_noise = self._compute_snoise()
        return (Xhat, yhat, s_noise, m_0, S_0)

        he = self._compute_he()
        minux = de * 1/he
        
        # yhat: transformed y
        yhat = self._compute_ytransform(Xhat, minux)
        
        # delete unused variables            
        del de, he, minux
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()   
        
        # Xhat nad Jacobians
        if is_lightweight:
            return (Xhat, yhat, None, None, None)
        
        # other model quantiles    
        m_0 = self._compute_m0()
        S_0 = self._compute_S0()
        s_noise = self._compute_snoise()
        return (Xhat, yhat, s_noise, m_0, S_0)
