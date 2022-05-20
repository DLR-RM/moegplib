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

from moegplib.networks.kittimodelquantiles import VisualOdometryModelQuantiles
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood

from moegplib.networks.modelquantiles import ModelQuantiles, DetectronQuantiles
from moegplib.moegp.gkernels import NTKMISO, ExactNTKGP
from moegplib.moegp.compression import JacobianPruner, JacobianPrunerVIO
from moegplib.moegp.activelearning import ActiveLearner
from moegplib.utils.metric import rmse_f, nll_f

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class ConquerStepBase:
    """ A parent class to patchwork gp.
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
        Computes experts nearby, and prepares the
        arguments for the patchwork gp.

        Args:
            expnum (int): expert number.
            gater (boundaries object): contains gating function variables.
        """
        # compute quantities of nearby experts
        self.experts = \
            self._cal_experts_and_nearby(expnum, gater.CONN, gater.bidx)
        
        # assigning the remiaining patchwork arguments
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

        return experts

    def boundary_activelearn(self, experts, expertnum, targetout, 
                             Xtrainhat, Ytrainhat, pruneidx=None):
        """Performs active learning on boundary experts

        Args:
            experts (list): contains boundary experts number and expert number i.
            targetout (int): output dimension.
            expertnum (int): specific expert number.
            pruneidx (torch.Tensor): pruning index.
            Xtrainhat (torch.Tensor): Jacobian of neural networks N x P.
            Ytrainhat (torch.Tensor): Transformed output N.

        Returns:
            Xtrainhat (torch.Tensor): Jacobian of neural networks K x P.
            Ytrainhat (torch.Tensor): Transformed output K.
        """
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
    
    def _cal_transform_advanced(self, targetout, sparsity=0.5,
                                savemode='zarr', is_test=False):
        """ loads the saved model quantiles or the transforms.

        Args:
            targetout (int): output dimension
            sparsity (float, optional): sparsity parameter raning 0.0-1.0 (0% to 100%). Defaults to 0.5.
            savemode (str, optional): choosing how to load model quantiles e.g. using 'zarr'. Defaults to 'zarr'.
            is_test (bool, optional): if yes, returns the test quantities. Defaults to False.

        Raises:
            NotImplementedError: Only supports zarr mode.
            NotImplementedError: Only supports zarr mode.
            AttributeError: Only supports zarr mode.

        Returns:
            Xtrainhat (torch.Tensor): N x P Jacobian of neural networks
            Ytrainhat (torch.Tensor): M transformed output
        """
        logging.info("Loading the model quantiles")
        logging.info("Training set")
        # call the central expert
        Xtrainhat = self.Jsaveload.load_zarr(nrexpert=self.expertnum,
                                             targetout=targetout,
                                             returnmode='Jtrain')
        Ytrainhat = self.Jsaveload.load_zarr(nrexpert=self.expertnum,
                                             targetout=targetout,
                                             returnmode='yhat')
        
        # return nan if expert does not have any data
        if Xtrainhat is np.nan:
            return np.nan, np.nan
        
        # if post-sparsity parameter is set.
        if float(sparsity) > 0.0 and Xtrainhat is not np.nan:
            # prune the central expert jacobian.
            logging.info("Expert %s has the amounut set %s", \
                str(self.expertnum), str(len(Xtrainhat)))
            logging.info("Pruning the central expert")
            pruner = JacobianPruner(sparsity=sparsity, mode='sparse')
            Xtrainhat, pruneidx = pruner.expertsprune(Xtrainhat, do_pruneindx=True)

            # performs active learning
            logging.info("Pruning and active learning of the boundary experts")
            if float(self.targetsize) > 0.0:
                Xtrainhat, Ytrainhat = self.boundary_activelearn(self.experts, 
                                                                 self.expertnum,
                                                                 targetout, 
                                                                 Xtrainhat,
                                                                 Ytrainhat,
                                                                 pruneidx)
            
            # save the jacobian indices
            expset = 'experts/' + 'target' + str(targetout)
            self.saveload.save_pickle(pruneidx, expset, self.expertnum)
            return Xtrainhat, Ytrainhat
        
        # if post-sparsity parameter is zero. 
        if float(sparsity) == 0.0 and Xtrainhat is not np.nan:
            # active learn without pruning
            logging.info("Expert %s has the amounut set %s", \
                str(self.expertnum), str(len(Xtrainhat)))
            logging.info("Active learning of the boundary experts")
            if float(self.targetsize) > 0.0:
                Xtrainhat, Ytrainhat = self.boundary_activelearn(self.experts,
                                                                    self.expertnum,
                                                                    targetout, 
                                                                    Xtrainhat,
                                                                    Ytrainhat)
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
            logger = self.Jsaveload.load_ckp(self.expertnum) # to be only ran afterwards

            # handling exceptions 
            if logger is np.nan:
                print("logger being nan?")
                return np.nan, np.nan
            
            # defining the training model
            likelihood = gpytorch.likelihoods.GaussianLikelihood() 
            gpmodel = ExactNTKGP(logger['Jtrain'], logger['yhat'], likelihood)
            
            # models for GPs
            return gpmodel.to(self.device), likelihood.to(self.device)
        else:
            # obtaining the reduced data
            Xtrainhat, Ytrainhat = self._cal_transform_advanced(targetout, sparsity=sparsity,
                                                                savemode='zarr', is_test=False)

            # return nan if expert does not have any data
            if Xtrainhat is np.nan:
                return np.nan, np.nan            
            return Xtrainhat, Ytrainhat


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
                Xtesthat = self._cal_transform_advanced(targetout=targetout, is_test=True)
                
                # pruning 
                if not sparsity == 0.0:
                    pruner = JacobianPruner(sparsity=sparsity, mode='sparse') 
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
