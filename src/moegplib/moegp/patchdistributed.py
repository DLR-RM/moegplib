""" This files contain patch gp method implementation incl basic gp from 1d to multiple dimensions.
"""
import logging
import torch
import itertools
import gpytorch
import gc
import numpy as np
import scipy.sparse as sparse
import os

from torch.multiprocessing import Queue, Pool, Process, set_start_method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from moegplib.networks.modelquantiles import ModelQuantiles
from moegplib.moegp.gkernels import NTKMISO, ExactNTKGP
from moegplib.moegp.compression import JacobianPruner
from moegplib.moegp.activelearning import ActiveLearner
from moegplib.moegp.base import ConquerStepNeuralTangent
from moegplib.utils.metric import rmse_f, nll_f

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class PatchDistributed(ConquerStepNeuralTangent):
    """Single input, but multiple output gater function.
    
    Args:
        ConquerStepNeuralTangent (python class): inherits the base class for patch gp
    """
    
    def __init__(self, outdim, nr_experts, sparsity, args, testdata,
                 gaterlist, Jsaveload, saveload, device='cpu', nr_experts_init=0, 
                 lr=0.1, training_iter=100, targetsize=0.3, initsize=0.5, n_queries=3,
                 qmethod='gp_regression', savemode=True, saver=None):
        """Initialization
        Args:
            testdata ([type]): [description]
            outdim ([type]): [description]
            nr_experts (int): total number of experts.
            sparsity (float, optional): sparsity parameter raning 0.0-1.0 (0% to 100%). Defaults to 0.0.
            gaterlist (list): list containing the gater object class
            Jsaveload ([type]): [description]
            saveload ([type]): [description]
            device (str, optional): 'cpu' or 'cuda'. Defaults to 'cpu'.
            nr_experts_init (int, optional): initial expert number (only for patchwork arguments). Defaults to 0.
            lr (float, optional): Learning rate for GP training. Defaults to 0.1.
            training_iter (int, optional): Total number of iterations per GP. Defaults to 100.
            targetsize (float, optional): Ratio of the target M < N. Defaults to 0.3.
            initsize (float, optional): Initial pool for the GP. Defaults to 0.5.
            n_queries (int, optional): Number of queries. Defaults to 3.
            qmethod (str, optional): Option to specify the queiry method. Defaults to 'gp_regression'.
            savemode (bool, optional): Only for the division step. Defaults to True.
        """
        super().__init__(nrexpert=nr_experts_init, gater=gaterlist[0], lr=lr, training_iter=training_iter,
                         targetsize=targetsize, initsize=initsize, n_queries=n_queries, qmethod=qmethod,
                         Jsaveload=Jsaveload, saveload=saveload, device=device, savemode=savemode)
        self.outdim = outdim
        self.nr_experts = nr_experts
        self.gaterlist = gaterlist
        self.sparsity = sparsity
        self.saveload = saveload
        self.lr = lr
        self.training_iter = training_iter # FIXME: double defining.
        self.args = args
        if args.pre_sparsity == 0.0:
            self.is_zeroout = False
        else:        
            self.is_zeroout = True
        if self.is_zeroout:
            if saver is not None:
                masks_file = self.args.checkpoint_dir + '/' + saver + 'mask_pruned_dnn.pth'
            else:
                masks_file = self.args.checkpoint_dir + 'mask_pruned_dnn.pth'
            self.neuralpruner = JacobianPruner(sparsity=None,
                                               masks_file=masks_file,
                                               mode='zeroesout')
        self.testdata = testdata
        self.results_lst = Queue()

    def __call__(self, expertn, gpu_id):
        # for expertnum in range(int(self.nr_experts)):
        print("\033[91m############### IN conquer step: expert ID {} obtains GPU {}! ###############\033[0m".format(expertn, gpu_id))
        torch.cuda.set_device(gpu_id)
        self.device = gpu_id
        self.Jsaveload.device = gpu_id
        mll, rmse = self.patchtrain(expertn)
        self.results_lst.put((expertn, rmse, mll))
        print("\033[91m############### IN conquer step: expert ID {} FNISHED on GPU {}! ###############\033[0m".format(expertn, gpu_id))
        return (expertn, rmse, mll)
    
    def patchtrain(self, expertnum, is_multigp=True, is_save=True):
        """Training the patchwork gp
        Args:
            is_multigp (bool, optional): If true, use multi-output GP. Defaults to False.
            is_save (bool, optional): If true, save the GP model. Defaults to True.
        Returns:
            mllsaver (dict): Contains the list of mll values per experts, dict of target dimensions. 
        """

        # check if any files (only the last dim)
        # if self.saveload.is_file('experts/' + 'target' + str(self.outdim-1), expertnum):
        #     continue

        # contruct the mutl-output gp for an expert
        gpmodellist, likelihoodlist = self._construct_multioutput(expertnum)

        # exceptions
        # if gpmodellist is np.nan or likelihoodlist is np.nan:
        #     continue
        
        # training intialization
        gpmodellist.train(), likelihoodlist.train()
        optimizer = torch.optim.Adam([
                        {'params': gpmodellist.parameters()},  # Includes all submodel and all likelihood parameters
                    ], lr=self.lr)
        mll = SumMarginalLogLikelihood(likelihoodlist, gpmodellist)
        
        # training loop
        with gpytorch.settings.cholesky_jitter(1e-5):
            for i in range(self.training_iter):
                optimizer.zero_grad()
                output = gpmodellist(*gpmodellist.train_inputs)
                loss = -mll(output, gpmodellist.train_targets)
                loss.backward(retain_graph=True)
                
                # logging
                noises = [gpmodellist.likelihood.likelihoods[out].noise.item() \
                    for out in range(int(self.outdim))]
                vars = [gpmodellist.models[out].covar_module.variance.item() \
                    for out in range(int(self.outdim))]
                if (i % 5==0):
                    logging.info('Expert number: %s', str(expertnum))
                    logging.info('Iter %d/%d - Loss: %.3f Parameter:' \
                        % (i + 1, self.training_iter, loss.item()))
                    logging.info('Avg. Noise: %s, Avg. Var: %s', 
                                    str(np.mean(np.asarray(noises))),
                                    str(np.mean(np.asarray(vars))))
                optimizer.step()
                
                # escape if noise is too small or Nan
                if loss.item() != loss.item() or any(t< 1e-4 for t in noises): 
                    continue
        
        # eval on train
        with torch.no_grad():
            idx_select = [len(self.gaterlist[out].idx[expertnum]) for out in range(int(self.outdim))]
            rmse = [rmse_f(likelihoodlist(*output)[out].mean.detach().cpu().numpy()[0:idx_select[out]],
                            gpmodellist.train_targets[out].detach().cpu().numpy()[0:idx_select[out]]) \
                                for out in range(int(self.outdim))]
            
        return loss.item(), rmse
    
    def _construct_multioutput(self, expertnum, is_test=False):
        """[summary]
        Args:
            expertnum ([type]): [description]
        Returns:
            [type]: [description]
        """
        # gp joint 1
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[0])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel1, likelihood1 = self.moegp(0, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel1 is np.nan or likelihood1 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat1, Ytrainhat1 \
                    = self.moegp(0, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat1 is np.nan:
                return np.nan, np.nan
            likelihood1 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel1 = ExactNTKGP(Xtrainhat1, Ytrainhat1, likelihood1)
        
        # gp joint 2
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[1])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel2, likelihood2 = self.moegp(1, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel2 is np.nan or likelihood2 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat2, Ytrainhat2 \
                    = self.moegp(1, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat2 is np.nan:
                return np.nan, np.nan
            likelihood2 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel2 = ExactNTKGP(Xtrainhat2, Ytrainhat2, likelihood2)
        
        # gp joint 3
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[2])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel3, likelihood3 = self.moegp(2, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel3 is np.nan or likelihood3 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat3, Ytrainhat3 \
                    = self.moegp(2, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat3 is np.nan:
                return np.nan, np.nan
            likelihood3 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel3 = ExactNTKGP(Xtrainhat3, Ytrainhat3, likelihood3)
        
        # gp joint 4
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[3])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel4, likelihood4 = self.moegp(3, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel4 is np.nan or likelihood4 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat4, Ytrainhat4 \
                    = self.moegp(3, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat4 is np.nan:
                return np.nan, np.nan
            likelihood4 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel4 = ExactNTKGP(Xtrainhat4, Ytrainhat4, likelihood4)

        # gp joint 5
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[4])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel5, likelihood5 = self.moegp(4, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel5 is np.nan or likelihood5 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat5, Ytrainhat5 \
                    = self.moegp(4, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat5 is np.nan:
                return np.nan, np.nan
            likelihood5 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel5 = ExactNTKGP(Xtrainhat5, Ytrainhat5, likelihood5)

        # gp joint 6
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[5])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel6, likelihood6 = self.moegp(5, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel6 is np.nan or likelihood6 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat6, Ytrainhat6 \
                    = self.moegp(5, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat6 is np.nan:
                return np.nan, np.nan
            likelihood6 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel6 = ExactNTKGP(Xtrainhat6, Ytrainhat6, likelihood6)

        # gp joint 7
        logging.info("Preparing the patching work")
        try:
            self.patching_preps(expertnum, self.gaterlist[6])
        except IndexError:
            return np.nan, np.nan
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel7, likelihood7 = self.moegp(6, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel7 is np.nan or likelihood7 is np.nan:
                return np.nan, np.nan
        else:
            Xtrainhat7, Ytrainhat7 \
                    = self.moegp(6, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat7 is np.nan:
                return np.nan, np.nan
            likelihood7 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel7 = ExactNTKGP(Xtrainhat7, Ytrainhat7, likelihood7)
        
        # multi-output gp
        gpmodellist = gpytorch.models.IndependentModelList(gpmodel1, gpmodel2, gpmodel3,
                                                           gpmodel4, gpmodel5, gpmodel6,
                                                           gpmodel7)
        likelihoodlist = gpytorch.likelihoods.LikelihoodList(gpmodel1.likelihood, gpmodel2.likelihood,
                                                             gpmodel3.likelihood, gpmodel4.likelihood,
                                                             gpmodel5.likelihood, gpmodel6.likelihood,
                                                             gpmodel7.likelihood)
        return gpmodellist.to(self.device), likelihoodlist.to(self.device)
