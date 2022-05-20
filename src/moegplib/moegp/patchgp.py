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
import detectron2.data.transforms as T
import copy

from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood

from moegplib.networks.modelquantiles import ModelQuantiles, DetectronQuantiles
from moegplib.moegp.gkernels import NTKMISO, ExactNTKGP
from moegplib.moegp.compression import JacobianPruner
from moegplib.moegp.activelearning import ActiveLearner
from moegplib.utils.metric import rmse_f, nll_f

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class ConquerStepNeuralTangent():
    """The conquer step of GP uncertainty in DL.
    """    
    def __init__(self, nrexpert, gater, Jsaveload, saveload, initsize=0.5, n_queries=3,
                 device='cpu', lr=0.1, training_iter=100, targetsize=0.3, 
                 qmethod='gp_regression', savemode=True):
        """ Initialization. Computes the related variables
        of surrounding experts for the expert number nrexpert. 
        Args will be used for NTK kernels.

        Args:
            nrexpert (int): the expert number.
            gater (class object): gater class with the results of the gating function.
            Jsaveload (class object): jacobain saving and loading utility function.
            saveload (class object): model saving and loading utility function.
            initsize (float): Initial pool for the GP.
            n_queries (int): number of queries for active learning
            device (string): cpu or gpu. 
            lr (float): Learning rate for GP training.
            training_iter (int): Total number of iterations per GP.
            targetsize (float): Ratio of the target M < N.
            qmethod (str): Option to specify the queiry method.
            savemode (bool, optional): Only for the division step. Defaults to True.
        """
        self.nrexpert = nrexpert
        self.gater = gater
        self.Jsaveload = Jsaveload
        self.saveload= saveload
        self.device = device
        if savemode: # compute nearby experts
            self.experts, self.bidx_e = \
                self._cal_experts_and_nearby(self.nrexpert, 
                                             self.gater.CONN, 
                                             self.gater.bidx)
        self.lr = lr
        self.training_iter = training_iter
        self.targetsize = targetsize
        self.initsize = initsize
        self.n_queries = n_queries
        self.qmethod = qmethod
        
    def patching_preps(self, expnum, gater):
        """Preparations for the patchwork.
        Requries to be ran before exporting the model.

        Args:
            expnum (int): expert number.
            gater (boundaries object): contains gating function variables.
        """
        # compute quantities of nearby experts
        self.experts, self.bidx_e = \
            self._cal_experts_and_nearby(expnum, 
                                         gater.CONN,
                                         gater.bidx)
        self.expertnum = expnum

    def _cal_experts_and_nearby(self, i, CONN, bidx):
        """Calculates the experts near by.

        Args:
            i (int): the expert number i.
            CONN (list): contains p x 2 with experts (1st p x 1) and their boundaries (2nd p x 1)
            bidx (list): indices for the boundary points.

        Returns:
            experts (list): contains boundary experts number and expert number i.
            bidx_e (list): indices for experts list.
        """
        # per experts, extract neighbors
        experts = list()

        # list neighborhoods
        if not len(CONN):
            lst = []
        else:
            # compute a list of relevant neighbors for expert i
            lst_a = [] if np.nonzero(CONN[0] == i)[0].size == 0 \
                else (np.nonzero(CONN[0] == i)[0]).astype(int)
            lst_b = [] if np.nonzero(CONN[1] == i)[0].size == 0 \
                else (np.nonzero(CONN[1] == i)[0]).astype(int)
            if len(lst_a) and len(lst_b):
                lst = [lst_a, lst_b] # note this is only for 2-D case!
            elif len(lst_a) and not len(lst_b):
                lst = [lst_a]
            elif not len(lst_a) and len(lst_b):
                lst = [lst_b]
            else:
                logging.info("WARNING: no neighbor for expert %s", str(i))
                lst = list()
            lst = sorted(list(itertools.chain(*lst)))

            # finding all experts list incl. neighbors        
            experts.append([i])
            experts.append(CONN[1][lst_a].tolist()) 
            experts.append(CONN[0][lst_b].tolist()) 
            experts = list(itertools.chain(*experts))
            experts = [x+1 for x in experts] 
            experts = list(filter(None, experts)) 
            experts = list(set(experts)) 
            experts = [x-1 for x in experts] 
            experts = sorted(experts) 
            bidx_e = None 

        return experts, bidx_e

    def _cal_patchworkarg(self, i, experts, bidx_e, CONN, idx, idx_t):
        """ This functions computes localized indices per boundary.
        We think of it as staring from 0 to N localized experts.
        In this way, computation does not involve heavy matrices.
        
        Args:
            i (int): the expert number i.
            experts (list): contains boundary experts number and expert number i.
            bidx_e (list): indices for experts list.
            CONN (list): contains p x 2 with experts (1st p x 1) and their boundaries (2nd p x 1)
            idx (list): m x n list containing indices, w/ m the number of experts, and n the indices. 
            idx_t (list): test set m x n list indices, w/ m the number of experts, and n the indices. 

        Returns:
            patchkwargs (dict): contains arguments to patchwork kernel.
            idx_ce (list): indice for cutting out only the expert data.
        """
        # initialization
        bidxe_e = list(np.arange(0, len(bidx_e)))
        CONN_e = [CONN[0][bidx_e], CONN[1][bidx_e]]
        idx_e, idx_t_e = list(), list()
        cnt, cnt_t = 0, 0
        lidx_e = np.arange(0, len([np.concatenate([idx[ei] \
            for ei in experts])][0]))
        lidx_t_e = np.arange(0, len([np.concatenate([idx_t[ei] \
            for ei in experts])][0]))
        
        # new list variables
        for ei in experts:
            idx_e.append(np.arange(cnt, cnt+len(idx[ei])))
            idx_t_e.append(np.arange(cnt_t, cnt_t+len(idx_t[ei])))
            if ei == i:
                idx_ce = np.arange(cnt_t, cnt_t+len(idx_t[ei]), 1)
            cnt, cnt_t = cnt + len(idx[ei]), cnt_t + len(idx_t[ei])
            lst_ae = np.where(CONN_e[0] == ei)[0]
            lst_be = np.where(CONN_e[1] == ei)[0]
            if len(lst_ae) != 0:
                np.put(CONN_e[0], lst_ae, \
                    np.where(np.asarray(experts) == ei)[0] * np.ones(len(lst_ae)))
            if len(lst_be) != 0:
                np.put(CONN_e[1], lst_be, \
                    np.where(np.asarray(experts) == ei)[0] * np.ones(len(lst_be)))

        # saving relevant variables for our kernel
        patchkwargs = {'bidx': bidxe_e, 'idx': idx_e, 
                       'idx_t': idx_t_e, 'CONN': CONN_e}

        return patchkwargs, idx_ce

    def _cal_transform(self, targetout, savemode='zarr', 
                       is_test=False, do_bndpatch=False):
        """ Calculates the transformed quantities.
        These are Jacobian matrices, and projected outputs.

        Args:
            targetout (int): output dimension
            savemode (str, optional): choosing how to load model quantiles e.g. using 'zarr'. Defaults to 'zarr'.
            is_test (bool, optional): if yes, returns the test quantities. Defaults to False.
            do_bndpatch (bool, optional): choose whether to use boundary points explicitly. Defaults to False.

        Returns:
            Xtrainhat (Torch.tensor): the jacobian matrix.
            Ytrainhat (Torch.tensor): the transformed output.
        """
        if savemode == 'zarr':            
            # Xtesthat for testing
            if is_test:
                # mode to test set
                Xtesthat = torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='Jtest') \
                                    for ei in self.experts])
                #self.arg['ntest'] = Xtesthat.shape[0]  
                return Xtesthat
            else:
                if do_bndpatch:
                    # Xtrainhat: combininig Jtrain and Jbnd                
                    Xtrainhat = torch.cat((torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='Jtrain') \
                                           for ei in self.experts]), 
                                           self.Jsaveload.load_ckp(nrexpert=self.expertnum, targetout=targetout)))
                    
                    # Ytrainhat: combining yhat and zeros (due to boundaries)
                    Ytrainhat = torch.cat((torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='yhat') \
                                           for ei in self.experts]), torch.zeros(len(self.bidx_e))))
                else:
                    # Xtrainhat: combininig Jtrain              
                    Xtrainhat = torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='Jtrain') \
                                           for ei in self.experts])
                    
                    # Ytrainhat: combining yhat
                    Ytrainhat = torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='yhat') \
                                           for ei in self.experts])
        elif savemode == 'npy':
            raise NotImplementedError
        elif savemode == 'ckp':
            raise NotImplementedError
        else:
            raise AttributeError
        
        return Xtrainhat, Ytrainhat
    
    def _cal_transform_advanced(self, targetout, sparsity=0.5,
                                savemode='zarr', is_test=False):
        """[summary]

        Args:
            targetout (int): output dimension
            sparsity (float, optional): sparsity parameter raning 0.0-1.0 (0% to 100%). Defaults to 0.5.
            savemode (str, optional): choosing how to load model quantiles e.g. using 'zarr'. Defaults to 'zarr'.
            is_test (bool, optional): if yes, returns the test quantities. Defaults to False.

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]
            AttributeError: [description]

        Returns:
            Xtrainhat (torch.Tensor): N x P Jacobian of neural networks
            Ytrainhat (torch.Tensor): M transformed output
        """
        logging.info("Loading the model quantiles")
        if savemode == 'zarr':
            if is_test:
                logging.info("Test set")
                if not sparsity == 0.0:
                    # load the pruning indices
                    expset = 'experts/' + 'target' + str(targetout)
                    pruneidx = self.saveload.load_pickle(expset, self.expertnum)
                    Xtesthat = torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, 
                                                                   returnmode='Jtest')[:, pruneidx] \
                                          for ei in self.experts])
                else:
                    Xtesthat = torch.cat([self.Jsaveload.load_zarr(nrexpert=ei, targetout=targetout, returnmode='Jtest') \
                                          for ei in self.experts])
                # if Xtest is not None:
                #     self.arg['ntest'] = Xtesthat.shape[0]
                return Xtesthat              
            else:
                logging.info("Training set")
                # call the central expert
                Xtrainhat = self.Jsaveload.load_zarr(nrexpert=self.expertnum,
                                                     targetout=targetout,
                                                     returnmode='Jtrain')
                Ytrainhat = self.Jsaveload.load_zarr(nrexpert=self.expertnum,
                                                     targetout=targetout,
                                                     returnmode='yhat')
                
                def boundary_activelearn(experts, expertnum, targetout, 
                                         Xtrainhat, Ytrainhat, pruneidx=None):
                    # call the boundary experts expert 
                    for ei in experts:
                        if not ei == expertnum:
                            logging.info("Active set for %s", str(ei))
                            Xtrain_ei = self.Jsaveload.load_zarr(nrexpert=ei,
                                                                targetout=targetout,
                                                                returnmode='Jtrain')
                            Ytrain_ei = self.Jsaveload.load_zarr(nrexpert=ei,
                                                                targetout=targetout,
                                                                returnmode='yhat')
                            if Xtrain_ei is not np.nan or Ytrain_ei is not np.nan:
                                # prune the boundary experts
                                if pruneidx is not None:
                                    Xtrain_ei = Xtrain_ei[:, pruneidx]
                                
                                # active learn
                                alearner = ActiveLearner(Xtrain_ei, Ytrain_ei, patchargs={}, targetsize=self.targetsize,
                                                        initsize=self.initsize, qmethod=self.qmethod, lr=self.lr,
                                                        training_iter=50, device=self.device)
                                Xtrain_ei, Ytrain_ei = alearner()
                                
                                # concentenate and remove
                                Xtrainhat = torch.cat((Xtrainhat, Xtrain_ei.detach().cpu()))
                                Ytrainhat = torch.cat((Ytrainhat, Ytrain_ei.detach().cpu()))
                                del Xtrain_ei, Ytrain_ei, alearner
                                gc.collect() # garbage collect
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                    
                    return Xtrainhat, Ytrainhat
                    
                if float(sparsity) > 0.0 and Xtrainhat is not np.nan:
                    # prune the central expert jacobian
                    logging.info("Expert %s has the amounut set %s", str(self.expertnum), str(len(Xtrainhat)))
                    logging.info("Pruning the central expert")
                    pruner = JacobianPruner(sparsity=sparsity, mode='sparse')
                    Xtrainhat, pruneidx = pruner.expertsprune(Xtrainhat, do_pruneindx=True)
                    logging.info("Pruning and active learning of the boundary experts")
                    if float(self.targetsize) > 0.0:
                        Xtrainhat, Ytrainhat = boundary_activelearn(self.experts, 
                                                                    self.expertnum,
                                                                    targetout, 
                                                                    Xtrainhat,
                                                                    Ytrainhat,
                                                                    pruneidx)
                    
                    # save the jacobian indices
                    expset = 'experts/' + 'target' + str(targetout)
                    self.saveload.save_pickle(pruneidx, expset, self.expertnum)
                if float(sparsity) == 0.0 and Xtrainhat is not np.nan:
                    # active learn without pruning
                    logging.info("Expert %s has the amounut set %s", str(self.expertnum), str(len(Xtrainhat)))
                    logging.info("Active learning of the boundary experts")
                    if float(self.targetsize) > 0.0:
                        Xtrainhat, Ytrainhat = boundary_activelearn(self.experts,
                                                                    self.expertnum,
                                                                    targetout, 
                                                                    Xtrainhat,
                                                                    Ytrainhat)
                if Xtrainhat is np.nan:
                    return np.nan, np.nan
        elif savemode == 'npy':
            raise NotImplementedError
        elif savemode == 'ckp':
            raise NotImplementedError
        else:
            raise AttributeError
        return Xtrainhat, Ytrainhat
    
    def moegp(self, targetout, sparsity=0.0, is_test=False,
              do_activelearn=True, is_multigp=False):
        """Mixtures of Experts GP. 

        Args:
            targetout (int): output dimension
            sparsity (float, optional): sparsity parameter raning 0.0-1.0 (0% to 100%). Defaults to 0.0.
            is_test (bool, optional): if true, then work with test variables. Defaults to False.
            do_activelearn (bool, optional): if yes, performs active learning. Defaults to True.
            is_multigp (bool, optional): if yes, only returns xhat and yhat. Defaults to False

        Returns:
            gpmodel (gpytorch model object): patchwork gaussian processes
            likelihood (gypytorch likelihood object): likelihood
            Xtrainhat (Torch.tensors): Jacobian of neural networks
            Ytrainhat (Torch.tensors): Transformed output
        """        
        # evaluation
        if is_test:
            # load the models data
            self.Jsaveload.mode = str(targetout) + "final/"
            logger = self.Jsaveload.load_ckp(self.expertnum)
            
            # handling exceptions 
            if logger is np.nan:
                if is_multigp:
                    return np.nan, np.nan
                else:
                    return np.nan, np.nan, np.nan
            
            # defining the training model
            likelihood = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel = ExactNTKGP(logger['Jtrain'], logger['yhat'], likelihood)
            
            # models for GPs
            if is_multigp:
                return gpmodel.to(self.device), likelihood.to(self.device)
            else:
                # load the input data
                Xtesthat = self._cal_transform_advanced(targetout=targetout,
                                                        is_test=True)
                
                # pruning (TODO: load and do it!)
                if not sparsity == 0.0:
                    pruner = JacobianPruner(sparsity=sparsity, mode='sparse') # TODO: we need to save the masks
                    Xtesthat = pruner.expertsprune(Xtesthat)
                del logger
                return gpmodel.to(self.device), likelihood.to(self.device), \
                    Xtesthat.to(self.device)
        else:
            if do_activelearn:
                # obtaining the reduced data
                Xtrainhat, Ytrainhat = self._cal_transform_advanced(targetout, sparsity=sparsity,
                                                                    savemode='zarr', is_test=False)
            else:
                # obtaining the data
                Xtrainhat, Ytrainhat = self._cal_transform(targetout=targetout, is_test=False)
                
                # jacobian pruning
                if not sparsity == 0.0 and Xtrainhat is not np.nan:
                    pruner = JacobianPruner(sparsity=sparsity, mode='sparse')
                    Xtrainhat = pruner.expertsprune(Xtrainhat)
            
            # models for GPs
            if is_multigp:
                return Xtrainhat, Ytrainhat
            if not is_multigp and Xtrainhat is not np.nan:
                likelihood = gpytorch.likelihoods.GaussianLikelihood() 
                gpmodel = ExactNTKGP(Xtrainhat, Ytrainhat, likelihood)
                return gpmodel.to(self.device), likelihood.to(self.device), \
                    Xtrainhat.to(self.device), Ytrainhat.to(self.device)
            if not is_multigp and Xtrainhat is np.nan:
                return np.nan, np.nan, np.nan, np.nan


class PatchSIMO(ConquerStepNeuralTangent):
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
            # check if any files (only the last dim)
            if self.saveload.is_file('experts/' + 'target' + str(self.outdim-1), expertnum):
                continue

            # contruct the mutl-output gp for an expert
            gpmodellist, likelihoodlist = self._construct_multioutput(expertnum)

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
            
                # saving the variables
                expname = 'output' + str(expertnum)
                mllsaver[expname] = loss.item()
                rmsesaver[expname] = rmse
                
                # save the checkpoint
                if is_save:
                    for out in range(int(self.outdim)):
                        expset = 'experts/' + 'target' + str(out)
                        self.saveload.save_gpytorch(gpmodellist.models[out], expset, expertnum)
                        self.Jsaveload.mode = str(out) + "final/"
                        self.Jsaveload.save_ckp(gpmodellist.train_inputs[out][0].detach().cpu(), # tuple
                                                None, gpmodellist.train_targets[out].detach().cpu(),
                                                None, nrexpert=expertnum)
            
            # delete the variables
            del gpmodellist, likelihoodlist, loss, optimizer, mll, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return mllsaver, rmsesaver
    
    def patcheval(self, model, gaterlist, normalize=False, 
                  ymean=None, ystd=None, is_multigp=True):
        """ Evaluation method of Patchwork GP.
        Differs from prediction, that it is for
        already used/pre-processed test sets.

        Args:
            model (pytorch model): DNN pytorch model.
            is_multigp (bool, optional): [description]. Defaults to False.
        """
        # init
        nllsaver, rmsesaver = {}, {}
        ypredlist, ygtlist, gpvarlist = {}, {}, {}
        self.gaterlist = gaterlist

        for expertnum in range(int(self.nr_experts)):
            # accumulate the ytest belong to us.
            logging.info("Working on the expert %s", str(expertnum))            
            
            # contruct the mutl-output gp for an expert
            gpmodellist, likelihoodlist = self._construct_multioutput(expertnum, is_test=True)

            # catching the exception
            if gpmodellist is np.nan or likelihoodlist is np.nan:
                continue
            
            # loading the check points
            [gpmodellist.models[out].load_state_dict( \
                self.saveload.load_gpytorch('experts/' + 'target' + str(out), expertnum)) \
                    for out in range(int(self.outdim))]
            
            # evalaute the model
            gpmodellist.eval(), likelihoodlist.eval()
            
            # computing test jacobians
            xhattest1, xtest1, ytest1 = self._test_jacobian(model, 0, expertnum)
            xhattest2, xtest2, ytest2 = self._test_jacobian(model, 1, expertnum)
            xhattest3, xtest3, ytest3  = self._test_jacobian(model, 2, expertnum)
            xhattest4, xtest4, ytest4  = self._test_jacobian(model, 3, expertnum)
            xhattest5, xtest5, ytest5  = self._test_jacobian(model, 4, expertnum)
            xhattest6, xtest6, ytest6  = self._test_jacobian(model, 5, expertnum)
            xhattest7, xtest7, ytest7  = self._test_jacobian(model, 6, expertnum)

            # chop the test data for fitting more train in memory?
            # then all test jacobian part goes to cpu, and move here to gpu.
            try: 
                set1size, set2size, set3size = xhattest1.shape[0], \
                    xhattest2.shape[0], xhattest3.shape[0]
                set4size, set5size, set6size, set7size = xhattest4.shape[0], \
                    xhattest5.shape[0], xhattest6.shape[0], xhattest7.shape[0]
            except AttributeError:
                continue
            allsize = set1size + set2size + set3size + \
                set4size + set5size + set6size + set7size
            splitnumber = 10
            nsplit1 = int(set1size/splitnumber)
            nsplit2 = int(set2size/splitnumber)
            nsplit3 = int(set3size/splitnumber)
            nsplit4 = int(set4size/splitnumber)
            nsplit5 = int(set5size/splitnumber)
            nsplit6 = int(set6size/splitnumber)
            nsplit7 = int(set7size/splitnumber)
            nsplitslist = [nsplit1, nsplit2, nsplit3, nsplit4, nsplit5, nsplit6, nsplit7]
            if not any(t is 0 for t in nsplitslist):
                # split the data into chunks
                logging.info("Splitting the data for evaluation.")
                xhattest1, xtest1, ytest1 = torch.split(xhattest1, nsplit1),\
                    torch.split(xtest1, nsplit1), torch.split(ytest1, nsplit1)
                xhattest2, xtest2, ytest2 = torch.split(xhattest2, nsplit2), \
                    torch.split(xtest2, nsplit2), torch.split(ytest2, nsplit2)
                xhattest3, xtest3, ytest3  = torch.split(xhattest3, nsplit3), \
                    torch.split(xtest3, nsplit3), torch.split(ytest3, nsplit3)
                xhattest4, xtest4, ytest4  = torch.split(xhattest4, nsplit4), \
                    torch.split(xtest4, nsplit4), torch.split(ytest4, nsplit4)
                xhattest5, xtest5, ytest5  = torch.split(xhattest5, nsplit5), \
                    torch.split(xtest5, nsplit5), torch.split(ytest5, nsplit5)
                xhattest6, xtest6, ytest6  = torch.split(xhattest6, nsplit6), \
                    torch.split(xtest6, nsplit6), torch.split(ytest6, nsplit6)
                xhattest7, xtest7, ytest7  = torch.split(xhattest7, nsplit7), \
                    torch.split(xtest7, nsplit7), torch.split(ytest7, nsplit7)
            else:
                logging.info("No splitting of the data for evaluation.")
                splitnumber = 1
                xhattest1, xtest1, ytest1 = torch.split(xhattest1, 1),\
                    torch.split(xtest1, 1), torch.split(ytest1, 1)
                xhattest2, xtest2, ytest2 = torch.split(xhattest2, 1),\
                    torch.split(xtest2, 1), torch.split(ytest2, 1)
                xhattest3, xtest3, ytest3  = torch.split(xhattest3, 1),\
                    torch.split(xtest3, 1), torch.split(ytest3, 1)
                xhattest4, xtest4, ytest4  = torch.split(xhattest4, 1),\
                    torch.split(xtest4, 1), torch.split(ytest4, 1)
                xhattest5, xtest5, ytest5  = torch.split(xhattest5, 1),\
                    torch.split(xtest5, 1), torch.split(ytest5, 1)
                xhattest6, xtest6, ytest6  = torch.split(xhattest6, 1),\
                    torch.split(xtest6, 1), torch.split(ytest6, 1)
                xhattest7, xtest7, ytest7  = torch.split(xhattest7, 1),\
                    torch.split(xtest7, 1), torch.split(ytest7, 1)

            # initialization
            nlllist = list()
            rmselist = list()

            for nsplited in range(0, splitnumber):
                # xtest and ytest into a list
                xtest = [xtest1[nsplited].to(self.device), xtest2[nsplited].to(self.device), xtest3[nsplited].to(self.device), \
                    xtest4[nsplited].to(self.device), xtest5[nsplited].to(self.device), xtest6[nsplited].to(self.device), xtest7[nsplited].to(self.device)]
                ytest = [ytest1[nsplited].to(self.device), ytest2[nsplited].to(self.device), ytest3[nsplited].to(self.device), \
                    ytest4[nsplited].to(self.device), ytest5[nsplited].to(self.device), ytest6[nsplited].to(self.device), ytest7[nsplited].to(self.device)]

                # cathcing the exceptions
                if any(t is None for t in xtest):
                    continue

                # make DNN predictions
                ypred = [self.neuralnetwork(model, xtest[out]).detach().cpu().numpy()\
                    for out in range(int(self.outdim))]

                # delete unused data points
                del xtest
                gc.collect() # garbage collect
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # making predictions
                with torch.no_grad():
                    # gaussian process uncertainty
                    with gpytorch.settings.max_eager_kernel_size(2), \
                        gpytorch.settings.fast_pred_var():
                            predictions = likelihoodlist(*gpmodellist(xhattest1[nsplited].to(self.device), 
                                                                      xhattest2[nsplited].to(self.device),
                                                                      xhattest3[nsplited].to(self.device), 
                                                                      xhattest4[nsplited].to(self.device),
                                                                      xhattest5[nsplited].to(self.device), 
                                                                      xhattest6[nsplited].to(self.device),
                                                                      xhattest7[nsplited].to(self.device)))

                    # evalaute nll and rmse
                    rmse = [rmse_f(ypred[out][:, out], 
                                ytest[out][:, out].cpu().numpy(), 
                                normalize, ymean[out], ystd[out]) for out in range(int(self.outdim))]
                    nll = [nll_f(ypred[out][:, out], 
                                ytest[out][:, out].detach().cpu().numpy(), 
                                pred.stddev.detach().cpu().numpy()**2, 
                                normalize, ymean[out], ystd[out]) for pred, out in zip(predictions, range(int(self.outdim)))]
                    rmselist.append(rmse)
                    nlllist.append(nll)

            # saving the final mll per output # FIXME: unnormalize, and see.
            expname = 'output' + str(expertnum)
            nllsaver[expname] = nlllist
            rmsesaver[expname] = rmselist
            
            # delete the variables
            del gpmodellist, likelihoodlist, rmse, nll
            del xhattest1, xhattest2, xhattest3, xhattest4, xhattest5, xhattest6, xhattest7
            del ytest1, ytest2, ytest3, ytest4, ytest5, ytest6, ytest7
            del ytest, ypred
            gc.collect() # garbage collect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return nllsaver, rmsesaver
    
    def _accumulate_test_set(self, nr_expert, dataset, bdata, targetout, 
                             batchsize=324, is_multigp=False):
        # initialization
        n2m_lower, n2m_upper = 0, batchsize
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=batchsize, 
                                                 shuffle=False, 
                                                 num_workers=4)
        
        # function to get test data per expert
        def testdata_per_experts(batchsize, dataloader,
                                 n2m_lower, n2m_upper, 
                                 expert_nr, bdata, do_maxbatch=False):
            """ test data per expert
            Uses idx_t list to selects the test elements
            that belong to a specific expert.
            """
            # initialization
            idx = bdata.idx_t[expert_nr]
            xtest, ytest = list(), list()
            
            # loop through the data
            for batch_ndx, sample in enumerate(dataloader):
                xtemp = sample['Xsample'].detach().cpu().numpy()
                ytemp = sample['Ysample'].detach().cpu().numpy()
                
                if do_maxbatch:
                    return xtemp[idx, :], ytemp[idx]
            
                # ranges of indicies
                index = [x if x>=n2m_lower and x<n2m_upper else None for x in idx]
                index = [org for org in index if org is not None] 
                index = list(set(index))
                index = [int(x - batch_ndx*batchsize) for x in index]
            
                # append the data
                if index:
                    xtest.append(xtemp[index, :])
                    ytest.append(ytemp[index])
            
                # update the lower and upper bounds
                n2m_lower += batchsize
                n2m_upper += batchsize
            return np.concatenate(xtest), np.concatenate(ytest)
        
        # save the test data per expert
        try:
            xtest, ytest = testdata_per_experts(batchsize, dataloader, 
                                                n2m_lower, n2m_upper,
                                                nr_expert, bdata)
        except ValueError:
            return None, None
        
        # returning the values
        if is_multigp:
            return torch.from_numpy(xtest).to(self.device),\
                torch.from_numpy(ytest).to(self.device)
        else:
            return torch.from_numpy(xtest).to(self.device),\
                torch.from_numpy(ytest[:, targetout]).to(self.device)
                
    def _test_jacobian(self, model, targetout, expertnum,
                       xtest=None, ytest=None):
        # accumulate test set
        if xtest is None and ytest is None:
            xtest, ytest = self._accumulate_test_set(expertnum, self.testdata, 
                                                     self.gaterlist[targetout],
                                                     targetout, is_multigp=True)
        
        # return zeros if there is no test set
        if xtest is None and ytest is None:
            return None, None, None
        
        # model quantiles (compute xhat).
        mq = ModelQuantiles(model=model,
                            data=(xtest, None), 
                            delta=1, 
                            outputdim=7, # FIXME: no hard typing
                            targetout=targetout,
                            devices='cpu')
        (Xhattest, _, _, _, _) = mq.projection(xtest)
        
        # neural pruning
        if self.is_zeroout:
            Xhattest_pruned = self.neuralpruner.neuralprune( \
                Xhattest.reshape((Xhattest.shape[1], Xhattest.shape[2])))
        else:
            Xhattest_pruned = Xhattest.reshape((Xhattest.shape[1], Xhattest.shape[2]))
        
        # pruning
        if not self.sparsity == 0.0:
            # load the pruning indices
            expset = 'experts/' + 'target' + str(targetout)
            pruneidx = self.saveload.load_pickle(expset, expertnum)
            return Xhattest_pruned[:, pruneidx[0]], xtest, ytest

        return Xhattest_pruned, xtest, ytest
    
    def _construct_multioutput(self, expertnum, is_test=False):
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

    def neuralnetwork(self, model, xtest, targetout=None):
        """Prediction method of neural network.

        Args:
            model (pytorch model): DNN model
            xtest (tensor): input tensor

        Returns:
            tensor: predictions of neural network.
        """
        model.eval() # TODO: move the model out?
        with torch.no_grad():
            ypred = model(xtest.double())
        if targetout is not None:
            return ypred[:, targetout]
        return ypred

