""" This files contain patch gp method implementation incl basic gp from 1d to multiple dimensions.
"""
import logging
import torch
import gpytorch
import gc
import numpy as np
import scipy.sparse as sparse
import os
import detectron2.data.transforms as T

from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood

from moegplib.networks.modelquantiles import ModelQuantiles, DetectronQuantiles
from moegplib.moegp.gkernels import NTKMISO, ExactNTKGP
from moegplib.moegp.compression import JacobianPruner
from moegplib.moegp.activelearning import ActiveLearner
from moegplib.utils.metric import rmse_f, nll_f
from moegplib.moegp.base import ConquerStepBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class PatchDetectron2(ConquerStepBase):
    """Single input, but multiple output gater function.
    
    Args:
        ConquerStepBase (python class): inherits the base class for patch gp
    """
    
    def __init__(self, nr_experts, args, gaterlist, Jsaveload, saveload, cfg, outdim=10, 
                 device='cpu', nr_experts_init=0, lr=0.1, training_iter=100, targetsize=0.3,
                 initsize=0.5, n_queries=3, qmethod='gp_regression', savemode=True, saver=None):
        """Initialization

        Args:
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
        self.sparsity = args.post_sparsity
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
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    def __call__(self):
        """ Prepares for the real time settings.
        """
        # initialization
        self.patchgp1, self.patchgp2 = list(), list()
        self.likelihood1, self.likelihood2 = list(), list()

        # looping through all the experts and load.
        for expertnum in range(int(self.nr_experts)):
            # accumulate the ytest belong to us.
            logging.info("Working on the expert %s", str(expertnum))            
            
            # contruct the mutl-output gp for an expert
            gpmodellist1, likelihoodlist1, gpmodellist2, likelihoodlist2 = \
                self._construct_multioutput(expertnum, is_test=True)

            if gpmodellist1 is np.nan or likelihoodlist1 is np.nan:
                continue

            if gpmodellist2 is np.nan or likelihoodlist2 is np.nan:
                continue

            # loading the check points
            [gpmodellist1.models[out+1].load_state_dict( \
                self.saveload.load_gpytorch('experts/' + 'target' + str(out), expertnum)) \
                    for out in range(0, 4)]
            # loading the check points
            [gpmodellist2.models[out+1].load_state_dict( \
                self.saveload.load_gpytorch('experts/' + 'target' + str(int(out+4)), expertnum)) \
                    for out in range(0, 4)]
            
            # evalaute the model
            gpmodellist1.eval(), likelihoodlist1.eval()
            gpmodellist2.eval(), likelihoodlist2.eval()

            # append to the list
            self.patchgp1.append(gpmodellist1)
            self.patchgp2.append(gpmodellist2)
            self.likelihood1.append(likelihoodlist1)
            self.likelihood2.append(likelihoodlist2)
    
    def patchtrain(self, is_multigp=True, is_save=True):
        """Training the patchwork gp

        Args:
            is_multigp (bool, optional): If true, use multi-output GP. Defaults to False.
            is_save (bool, optional): If true, save the GP model. Defaults to True.

        Returns:
            mllsaver (dict): Contains the list of mll values per experts, dict of target dimensions. 
        """
        # init
        mllsaver, rmsesaver = {}, {}
        
        for expertnum in range(int(self.nr_experts)):
            # classification - do not need training
            gpmodellist_c, likelihoodlist_c = \
                self._construct_multioutput(expertnum, is_test=False, is_class=True)

            # exceptions
            if gpmodellist_c is np.nan or likelihoodlist_c is np.nan:
                continue

            # save the checkpoint
            if is_save:
                counter = 0
                for out in range(8, 10):
                    expset = 'experts/' + 'target' + str(out)
                    self.saveload.save_gpytorch(gpmodellist_c.models[counter], expset, expertnum)
                    self.Jsaveload.mode = str(out) + "final/"
                    self.Jsaveload.save_ckp(gpmodellist_c.train_inputs[counter][0].detach().cpu(),
                                            None, gpmodellist_c.train_targets[counter].detach().cpu(),
                                            None, nrexpert=expertnum)
                    counter += 1
            del gpmodellist_c, likelihoodlist_c
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # regression - contruct the mutl-output gp for an expert
            gpmodellist, likelihoodlist = self._construct_multioutput(expertnum, is_test=False)

            # exceptions
            if gpmodellist is np.nan or likelihoodlist is np.nan:
                continue
            
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
                    noises_c1 = [gpmodellist.likelihood.likelihoods[out].noise.item() \
                        for out in range(0, 4)]
                    vars_c1 = [gpmodellist.models[out].covar_module.variance.item() \
                        for out in range(0, 4)]
                    noises_c2 = [gpmodellist.likelihood.likelihoods[out].noise.item() \
                        for out in range(4, 8)]
                    vars_c2 = [gpmodellist.models[out].covar_module.variance.item() \
                        for out in range(4, 8)]
                    if (i % 5==0):
                        logging.info('Expert number: %s', str(expertnum))
                        logging.info('Iter %d/%d - Loss: %.3f Parameter:' \
                            % (i + 1, self.training_iter, loss.item()))
                        logging.info('Avg. Noise for C1: %s, Avg. Var: %s', 
                                        str(np.mean(np.asarray(noises_c1))),
                                        str(np.mean(np.asarray(vars_c1))))
                        logging.info('Avg. Noise for C2: %s, Avg. Var: %s', 
                                        str(np.mean(np.asarray(noises_c2))),
                                        str(np.mean(np.asarray(vars_c2))))
                    optimizer.step()
                    
                    # escape if noise is too small or Nan
                    if loss.item() != loss.item() or any(t< 1e-4 for t in noises_c1): 
                        continue
                    if loss.item() != loss.item() or any(t< 1e-4 for t in noises_c2): 
                        continue
                
            # save the checkpoint
            if is_save:
                # regression
                for out in range(0, 8):
                    expset = 'experts/' + 'target' + str(out)
                    self.saveload.save_gpytorch(gpmodellist.models[out], expset, expertnum)
                    self.Jsaveload.mode = str(out) + "final/"
                    self.Jsaveload.save_ckp(gpmodellist.train_inputs[out][0].detach().cpu(),
                                            None, gpmodellist.train_targets[out].detach().cpu(),
                                            None, nrexpert=expertnum)
            
            # delete the variables
            del gpmodellist, likelihoodlist, loss, optimizer, mll, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _construct_multioutput(self, expertnum, is_test=False, is_class=False):
        # gp classification 0 NOTE: tensors into python default format e.g. list or dict cause mem problems.
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[0])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel1, likelihood1 = self.moegp(8, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel1 is np.nan or likelihood1 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat1, Ytrainhat1 \
                    = self.moegp(0, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat1 is np.nan:
                return np.nan, np.nan
            likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
            gpmodel1 = ExactNTKGP(Xtrainhat1, Ytrainhat1, likelihood1)
        
        # gp bounding box 1
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[1])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel2, likelihood2 = self.moegp(0, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel2 is np.nan or likelihood2 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat2, Ytrainhat2 \
                    = self.moegp(1, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat2 is np.nan:
                return np.nan, np.nan
            likelihood2 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel2 = ExactNTKGP(Xtrainhat2, Ytrainhat2, likelihood2)
        
        # gp bounding box 2
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[2])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel3, likelihood3 = self.moegp(1, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel3 is np.nan or likelihood3 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat3, Ytrainhat3 \
                    = self.moegp(2, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat3 is np.nan:
                return np.nan, np.nan
            likelihood3 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel3 = ExactNTKGP(Xtrainhat3, Ytrainhat3, likelihood3)
        
        # gp bounding box 3
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[3])

        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel4, likelihood4 = self.moegp(2, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel4 is np.nan or likelihood4 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat4, Ytrainhat4 \
                    = self.moegp(3, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat4 is np.nan:
                return np.nan, np.nan
            likelihood4 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel4 = ExactNTKGP(Xtrainhat4, Ytrainhat4, likelihood4)

        # gp bounding box 4
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[4])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel5, likelihood5 = self.moegp(3, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel5 is np.nan or likelihood5 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat5, Ytrainhat5 \
                    = self.moegp(4, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat5 is np.nan:
                return np.nan, np.nan
            likelihood5 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel5 = ExactNTKGP(Xtrainhat5, Ytrainhat5, likelihood5)

        # gp classificaiton 1
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[5])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel6, likelihood6 = self.moegp(9, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel6 is np.nan or likelihood6 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat6, Ytrainhat6 \
                    = self.moegp(5, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat6 is np.nan:
                return np.nan, np.nan
            likelihood6 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel6 = ExactNTKGP(Xtrainhat6, Ytrainhat6, likelihood6)

        # gp bounding box 1
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[6])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel7, likelihood7 = self.moegp(4, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel7 is np.nan or likelihood7 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat7, Ytrainhat7 \
                    = self.moegp(6, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat7 is np.nan:
                return np.nan, np.nan
            likelihood7 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel7 = ExactNTKGP(Xtrainhat7, Ytrainhat7, likelihood7)

        # gp bounding box 2
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[7])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel8, likelihood8 = self.moegp(5, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel8 is np.nan or likelihood8 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat8, Ytrainhat8 \
                    = self.moegp(7, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat8 is np.nan:
                return np.nan, np.nan
            likelihood8 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel8 = ExactNTKGP(Xtrainhat8, Ytrainhat8, likelihood8)

        # gp bounding box 3
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[8])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel9, likelihood9 = self.moegp(6, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel9 is np.nan or likelihood9 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat9, Ytrainhat9 \
                    = self.moegp(8, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat9 is np.nan:
                return np.nan, np.nan
            likelihood9 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel9 = ExactNTKGP(Xtrainhat9, Ytrainhat9, likelihood9)

        # gp bounding box 4
        logging.info("Preparing the patching work")
        self.patching_preps(expertnum, self.gaterlist[9])
        logging.info("Defining the GP model and likelihood")
        if is_test:
            gpmodel10, likelihood10 = self.moegp(7, sparsity=self.sparsity, is_multigp=True, is_test=True)
            if gpmodel10 is np.nan or likelihood10 is np.nan:
                return np.nan, np.nan, np.nan, np.nan
        else:
            Xtrainhat10, Ytrainhat10 \
                    = self.moegp(9, sparsity=self.sparsity, is_multigp=True)
            if Xtrainhat10 is np.nan:
                return np.nan, np.nan
            likelihood10 = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel10 = ExactNTKGP(Xtrainhat10, Ytrainhat10, likelihood10)
        
        # multioutput-gp for testing (regression)
        if is_test:
            gpmodellist1 = gpytorch.models.IndependentModelList(gpmodel1, gpmodel2, gpmodel3, gpmodel4, gpmodel5)
            gpmodellist2 = gpytorch.models.IndependentModelList(gpmodel6, gpmodel7, gpmodel8, gpmodel9, gpmodel10)
            likelihoodlist1 = gpytorch.likelihoods.LikelihoodList(gpmodel1.likelihood, gpmodel2.likelihood,
                                                                  gpmodel3.likelihood, gpmodel4.likelihood,
                                                                  gpmodel5.likelihood)
            likelihoodlist2 = gpytorch.likelihoods.LikelihoodList(gpmodel6.likelihood, gpmodel7.likelihood,
                                                                  gpmodel8.likelihood, gpmodel9.likelihood,
                                                                  gpmodel10.likelihood)
            return gpmodellist1.to(self.device), likelihoodlist1.to(self.device), \
                gpmodellist2.to(self.device), likelihoodlist2.to(self.device)

        # multioutput-gp for training (classification)
        elif not is_test and is_class:
            gpmodellist = gpytorch.models.IndependentModelList(gpmodel1, gpmodel6)
            likelihoodlist = gpytorch.likelihoods.LikelihoodList(gpmodel1.likelihood, gpmodel6.likelihood)
            return gpmodellist.to(self.device), likelihoodlist.to(self.device)

        # multioutput-gp for training (regression)
        else:
            gpmodellist = gpytorch.models.IndependentModelList(gpmodel2, gpmodel3, gpmodel4, gpmodel5,
                                                               gpmodel7, gpmodel8, gpmodel9, gpmodel10)
            likelihoodlist = gpytorch.likelihoods.LikelihoodList(gpmodel2.likelihood, gpmodel3.likelihood,
                                                                 gpmodel4.likelihood, gpmodel5.likelihood,
                                                                 gpmodel7.likelihood, gpmodel8.likelihood,
                                                                 gpmodel9.likelihood, gpmodel10.likelihood)
            return gpmodellist.to(self.device), likelihoodlist.to(self.device)
                
    def _test_jacobian(self, model, targetout, expertnum, sample, prejacobian=False):
        # model quantiles (compute xhat).
        if targetout==0:
            task="classification"
            classout=0 
            regout=0
        elif targetout==5:
            task="classification"
            classout=1 
            regout=0
        elif targetout>0 and targetout<5:
            task="regression"
            classout=0
            regout=targetout
        elif targetout>5 and targetout<10:
            task="regression"
            classout=1
            regout=targetout-6
        else:
            raise AttributeError
            
        mq = DetectronQuantiles(model=model,
                                data=None,
                                classout=classout,
                                regout=regout, 
                                task=task,
                                devices=self.device)
        (Xhattest, _, _, _, _) = mq.projection(sample)
        print(Xhattest)

        #if prejacobian: # return without pruning
        return Xhattest
        
    def patchworkgp(self, objectnr, expertnr, xhattest1, xhattest2, xhattest3, xhattest4, xhattest5):
        """Prediction method of Patchwork GP.

        Raises:
            NotImplementedError: [description]
        """
        # compute uncertainty
        with torch.no_grad():
            # gaussian process uncertainty
            with gpytorch.settings.max_eager_kernel_size(2), \
                gpytorch.settings.fast_pred_var():
                    if objectnr == 0:
                        predictions = self.likelihood1[expertnr](\
                            *self.patchgp1[expertnr](xhattest1, xhattest2, xhattest3, xhattest4, xhattest5))
                    elif objectnr == 1:
                        predictions = self.likelihood2[expertnr](\
                            *self.patchgp2[expertnr](xhattest1, xhattest2, xhattest3, xhattest4, xhattest5))
                    else:
                        raise AttributeError
        return predictions
