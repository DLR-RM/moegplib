"""A file that contains active learning method.
"""
import torch
import gpytorch
import logging
import gc
import copy
import numpy as np
import GPUtil

from gpytorch.mlls import ExactMarginalLogLikelihood
from moegplib.moegp.gkernels import NTKMISO, ExactNTKGP

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class ActiveLearner:
    def __init__(self, xtrain, ytrain, patchargs, device='cpu', targetsize=0.3,
                 initsize=0.5, n_queries=3, qmethod='gp_regression', lr=0.1, 
                 training_iter=100):
        """Active learner for Gaussian Processes

        Args:
            xtrain (torch.Tensor): N x P Jacobian matrix.
            ytrain (torch.Tensor): N transformed output.
            patchargs (dict): argument dictionary for patchGP.
            device (str, optional): 'cpu' or 'cuda'. Defaults to 'cpu'.
            targetsize (float, optional): Ratio of the target M < N. Defaults to 0.3.
            initsize (float, optional): Initial pool for the GP. Defaults to 0.5.
            n_queries (int, optional): Number of queries. Defaults to 3.
            qmethod (str, optional): Option to specify the queiry method. Defaults to 'gp_regression'.
            lr (float, optional): Learning rate for GP training. Defaults to 0.1.
            training_iter (int, optional): Total number of iterations per GP. Defaults to 100.
        """
        # initialization
        logging.info("Initializing the active learning")
        self.n_xtrain = int(len(xtrain))
        # memory reduction
        if self.n_xtrain > 1000:
            targetsize = 0.1
        if self.n_xtrain < 100:
            targetsize = 0.9
        self.targetsize = int(self.n_xtrain*targetsize)
        self.initsize = int(self.targetsize*initsize)
        self.n_queries = n_queries
        self.stepsize = int((self.targetsize-self.initsize)/self.n_queries)
        self.qmethod = qmethod
        self.device = device
        self.patchargs = patchargs
        self.lr = lr
        self.training_iter = training_iter
        logging.info("Target size %s out of total train size %s", str(self.targetsize), str(self.n_xtrain))
        
        # division of data to pool and train
        logging.info("Division of data to pool and init set")
        if self.targetsize==self.n_xtrain or self.targetsize==0.0:
            self.xtrain = xtrain
            self.ytrain = ytrain
        else:
            initial_idx = np.random.choice(range(self.n_xtrain), size=self.initsize, replace=False)
            self.xtrainpool = self._delete_train_data(xtrain, initial_idx, dim=0).detach().cpu().numpy()
            self.ytrainpool = self._delete_train_data(ytrain, initial_idx, dim=0).detach().cpu().numpy()
            self.xtrain = xtrain[initial_idx].detach().cpu().numpy()
            self.ytrain = ytrain[initial_idx].detach().cpu().numpy()
    
    def __call__(self, randopt=True):
        """Performs active learning

        Args:
            randopt (bool, optional): If yes, then we see to train with random query. Defaults to True.

        Returns:
            newxtrain (torch.Tensor): New M x P Jacobian with M < N
            newytrain (torch.Tensor): New M transformed output
        """
        if self.targetsize==self.n_xtrain or self.targetsize==0.0:
            return self.xtrain, self.ytrain
        if self.qmethod == 'random':
            # moving back to the device
            xtrain = torch.from_numpy(self.xtrain).to(self.device)
            ytrain = torch.from_numpy(self.ytrain).to(self.device)
            del self.xtrain, self.ytrain
            gc.collect() # garbage collect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # random query
            logging.info("Random query selected")
            gpmodel, likelihood = self._model_init(xtrain, ytrain)
            query_idx = self._queries(gpmodel, likelihood, self.xtrainpool)
            if query_idx is np.nan:
                return self.xtrain, self.ytrain
            
            # add train data (output the new data)
            newxtrain, newytrain = self._add_train_data(\
                self.xtrainpool[query_idx[0:int(self.stepsize*self.n_queries)]],
                self.ytrainpool[query_idx[0:int(self.stepsize*self.n_queries)]],
                xtrain.detach().cpu().numpy(),
                ytrain.detach().cpu().numpy())
            newxtrain = torch.from_numpy(newxtrain).to(self.device)
            newytrain = torch.from_numpy(newytrain).to(self.device)
            
            # optionally (train)
            if randopt:
                gpmodel, likelihood = self._model_init(newxtrain, newytrain)
                gpmodel, likelihood = self._teach(gpmodel, likelihood, 
                                                 newxtrain, newytrain,
                                                 lr=self.lr, 
                                                 training_iter=self.training_iter)
            logging.info("Set reduced from %s to %s", str(self.n_xtrain), str(len(newxtrain)))
            return newxtrain, newytrain
        else:
            logging.info("Uncertainty-based query selected")
            # moving back to the device
            xtrain = torch.from_numpy(copy.deepcopy(self.xtrain)).to(self.device)
            ytrain = torch.from_numpy(copy.deepcopy(self.ytrain)).to(self.device)
            
            # clear up
            del self.xtrain, self.ytrain
            gc.collect() # garbage collect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            for queries in range(int(self.n_queries+1)):                
                # model initialization
                gpmodel, likelihood = self._model_init(xtrain, ytrain)
                gpmodel, likelihood = self._teach(gpmodel, likelihood, 
                                                  xtrain, ytrain,
                                                  lr=self.lr, 
                                                  training_iter=self.training_iter)
                
                with torch.no_grad():
                    # query the indices
                    xtrainpool = torch.from_numpy(copy.deepcopy(self.xtrainpool)).to(self.device)
                    query_idx = self._queries(gpmodel, likelihood, xtrainpool)

                    if query_idx is np.nan:
                        break
                    
                    # add the data, and delete the pools
                    newxtrain, newytrain = self._add_train_data(\
                        self.xtrainpool[query_idx[-self.stepsize:]],
                        self.ytrainpool[query_idx[-self.stepsize:]],
                        xtrain.detach().cpu().numpy(),
                        ytrain.detach().cpu().numpy())
                    self._delete_pool_data(query_idx, self.stepsize)
                    
                    # clear up
                    del xtrainpool, gpmodel, likelihood, xtrain, ytrain
                    gc.collect() # garbage collect
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # for the loop to be valid
                    xtrain = torch.from_numpy(copy.deepcopy(newxtrain)).to(self.device)
                    ytrain = torch.from_numpy(copy.deepcopy(newytrain)).to(self.device)
                    
                    # clear up
                    del newxtrain, newytrain
                    gc.collect() # garbage collect
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            logging.info("Set reduced from %s to %s", str(self.n_xtrain), str(len(xtrain)))          
            return xtrain, ytrain
        
    def _queries(self, gpmodel, likelihood, xtrainpool):
        """Queries implementation

        Args:
            gpmodel ([type]): [description]
            likelihood ([type]): [description]
            xtrainpool ([type]): [description]

        Raises:
            NotImplementedError: [description]
            AttributeError: [description]

        Returns:
            [type]: [description]
        """
        # query methods
        if self.qmethod == 'gp_regression':
            # models to evaluation
            gpmodel.eval(), likelihood.eval()
            try:
                # obtain uncertainty
                with gpytorch.settings.max_eager_kernel_size(2), \
                    gpytorch.settings.fast_pred_var():
                        observed_pred = likelihood(gpmodel(xtrainpool, **self.patchargs))
                        sig2 = observed_pred.stddev **2
                
                # rank the samples according to the uncertainty
                query_idx = sig2.detach().cpu().numpy().argsort()[::-1].tolist()
            except RuntimeError:
                query_idx = np.nan
        elif self.qmethod == 'random':
            # random selection of points.
            query_idx = np.random.choice(range(len(xtrainpool)), size=len(xtrainpool), replace=False)
        elif self.qmethod == 'uncertainty_sampling':
            # uncertainty sampling (for classification)
            raise NotImplementedError
        else:
            raise AttributeError
        
        return query_idx
    
    def _teach(self, gpmodel, likelihood, xtrain, ytrain, lr=0.5, 
               training_iter=100, logdur=10):
        """Teach GP.

        Args:
            gpmodel ([type]): [description]
            likelihood ([type]): [description]
            xtrain ([type]): [description]
            ytrain ([type]): [description]
            lr (float, optional): [description]. Defaults to 0.5.
            training_iter (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]
        """
        # training intialization
        likelihood.to(self.device), gpmodel.to(self.device)
        gpmodel.train(), likelihood.train()
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, gpmodel)
        
        with gpytorch.settings.cholesky_jitter(1e-5):
            for i in range(training_iter):
                # output
                optimizer.zero_grad()
                output = gpmodel(xtrain, **self.patchargs)
                
                # loss
                loss = -mll(output, ytrain)
                loss.backward(retain_graph=True)
                if (i % logdur==0):
                    logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' \
                        % (i + 1, training_iter, loss.item(),
                        gpmodel.covar_module.variance.item(),
                        gpmodel.likelihood.noise.item()))
                optimizer.step()
                
                # escape if noise is too small
                if gpmodel.likelihood.noise.item() < 1e-4 or loss.item() != loss.item():
                    break
        
        # deleting variables
        del loss, mll, optimizer, output, ytrain, xtrain
        gc.collect() # garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return gpmodel, likelihood
    
    def _model_init(self, xtrain, ytrain):
        """Initiate the model and likelihood

        Args:
            xtrain (torch.Tensor): Jacobian matrix
            ytrain (torch.Tensor): Transformed output

        Returns:
            gpmodel (gpytorch.model): gp model instance
            likelihood (gpytorch.likelihood): gaussian likelihood instance.
        """
        # models for GPs
        likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        gpmodel = ExactNTKGP(xtrain, ytrain, likelihood)
        return gpmodel, likelihood
        
    def _delete_pool_data(self, query_idx, stepsize, mode = 'npy'):
        """[summary]

        Args:
            query_idx ([type]): [description]
            stepsize ([type]): [description]
            mode (str, optional): [description]. Defaults to 'npy'.

        Raises:
            AttributeError: [description]
        """
        # query_idx in good ones first! 
        if mode == 'npy':
            self.xtrainpool = self.xtrainpool[query_idx[-stepsize:]]
            self.ytrainpool = self.ytrainpool[query_idx[-stepsize:]]
        elif mode == 'tensor':
            self.xtrainpool = self.xtrainpool[query_idx[-stepsize:]].detach()
            self.ytrainpool = self.ytrainpool[query_idx[-stepsize:]].detach()
        else:
            raise AttributeError
    
    def _add_train_data(self, xselect, yselect, xtrain, ytrain, mode='npy'):
        """[summary]

        Args:
            xselect ([type]): [description]
            yselect ([type]): [description]
            xtrain ([type]): [description]
            ytrain ([type]): [description]
            mode (str, optional): [description]. Defaults to 'npy'.

        Raises:
            AttributeError: [description]

        Returns:
            [type]: [description]
        """
        if mode == 'npy':
            newxtrain = np.concatenate((xtrain, xselect))
            newytrain = np.concatenate((ytrain, yselect))
        elif mode == 'torch':
            newxtrain = torch.cat((xtrain, xselect)).detach()
            newytrain = torch.cat((ytrain, yselect)).detach()
        else:
            raise AttributeError
        return newxtrain, newytrain
    
    def _delete_train_data(self, arr, ind, dim=0):
        """Delete the training data (arr) given ind and axis.

        Args:
            arr (torch.Tensor): [description]
            ind (int): [description]
            dim (int): [description]

        Returns:
            [type]: [description]
        """
        arr = np.delete(arr.detach().cpu().numpy(), ind, dim)
        return torch.from_numpy(arr)

