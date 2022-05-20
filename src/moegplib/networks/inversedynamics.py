""" network.py contains scripts that define the neural network.
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
from moegplib.baselines.propagation import UncertaintyPropagator

torch.set_default_dtype(torch.double)


class SarcosPrimeNet(torch.nn.Module):
    def __init__(self, D_in=21, H=200, D_out=7):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SarcosPrimeNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a1 = self.linear1(x).clamp(min=0)
        a2 = self.linear2(a1).clamp(min=0)
        a3 = self.linear3(a2).clamp(min=0)
        a4 = self.linear4(a3).clamp(min=0)
        y_pred = self.linear5(a4)
        return y_pred


def SarcosPrimeSnet(D_in=21, H=200, D_out=7):
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),)
    return model


def get_model(weight_path: str = "", cuda: bool = False):
    """The model used for both SARCOS and KUKA experiments.
    
    Args:
        weight_path: Path to pre-trained weights. Returns untrained model if empty.
        cuda: Set to `True` if model was pre-trained on a GPU.

    Returns:
        PyTorch sequential model.
    """
    d_in = 21
    h = 200
    d_out = 7
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d_out))
    if weight_path:
        # models have a Dropout layer which is missing
        # in this model, so we need to renaim the last layer in
        # order to load his trained weights.
        try:
            model.load_state_dict(torch.load(weight_path,
                                             map_location=torch.device('cuda' if cuda else 'cpu')))
        except RuntimeError:
            state_dict = torch.load(weight_path,
                                    map_location=torch.device('cuda' if cuda else 'cpu'))["state_dict"]
            state_dict["8.weight"] = state_dict.pop("9.weight")
            state_dict["8.bias"] = state_dict.pop("9.bias")
            model.load_state_dict(state_dict)
    return model


class build_sarcos_net():
    def __init__(self, 
                dataset, 
                n_hidden, 
                save_model=False, 
                train_model=True,
                ckp_dir="", 
                n_epochs = 40, 
                tau = 1.0, 
                dropout = 0.05, 
                debug=False, 
                lr=0.001, 
                batch_size=128, 
                opt="Adam",
                normalize=True, 
                activation="relu", 
                cuda=True):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        self.dataset = dataset
        self.save_model = save_model
        self.ckp_dir = ckp_dir
        self.cuda = cuda
        if self.save_model:
            assert ckp_dir != "", "Save_model is {}, ckp_dir({}) should not be ''".format(save_model, ckp_dir)
        self.ckp_dir = ckp_dir
        
        # nework params
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.tanh()
        else:
            raise Exception("Only relu and tanh for activation are supported!")
        self.input_dim = dataset.X_train.shape[1]
        self.output_dim = dataset.y_train.shape[1]
        self.dropout = dropout
        self.n_hidden = n_hidden
        self._build_network()

        # original init is kaiming_uniform, to do sanity 
        # check with keras imple, switch to xavier uniform here:
        self._init_weights_xavier_uniform() 
            
        # training params
        self.tau = tau
        N = dataset.X_train.shape[0]
        lengthscale = 1e-2
        reg = lengthscale**2 * (1 - self.dropout) / (2. * N * tau)
        if opt == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)
        elif opt == "SGD":
            momentum = 0.9
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        criterion = torch.nn.MSELoss(reduction='sum')
        if train_model:
            self._train_network(optimizer, criterion, n_epochs, batch_size)
    
    def _init_weights_xavier_uniform(self):
        for m in self.model:
            if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def _build_network(self):
        # function for constructing a model with torch.nn.Sequential()
        layer_list = []
        layer_list.append(torch.nn.Linear(self.input_dim, self.n_hidden[0]))
        layer_list.append(self.activation)

        for i in range(len(self.n_hidden) - 1):
            layer_list.extend([torch.nn.Linear(self.n_hidden[i], self.n_hidden[i+1])])
            layer_list.append(self.activation)
        
        layer_list.append(torch.nn.Dropout(self.dropout))
        layer_list.append(torch.nn.Linear(self.n_hidden[-1], self.output_dim))
        self.model = torch.nn.Sequential(*layer_list).double()
        if self.cuda:
            self.model.cuda()
        
    def _train_network(self, optimizer, criterion, n_epochs, batch_size):
        if self.save_model:
            # define the check point class
            logger = SaveAndLoad(checkpth=self.ckp_dir)
            board = BoardLogger(checkpth=self.ckp_dir)
        
        # construc dataloader
        num_workers = 0 if self.dataset.cuda else 8
        dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=num_workers)
        for e in range(n_epochs):
            if e%100 == 0:
                if e > 0:
                    start=0
                    time_100ep=0
                    time_100ep += time.time() - start
                else:
                    time_100ep = 0.
                start = time.time()

            for i_batch, sample_batched in enumerate(dataloader):
                # taking the sample batch
                x_batch = sample_batched['Xsample']
                y_batch = sample_batched['Ysample']
                if not self.dataset.cuda and self.cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                y_pred = self.model(x_batch)

                # Compute and print loss
                loss = criterion(y_pred, y_batch)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if e == n_epochs - 1:
                if self.save_model:
                    print("Saving model at batch {}".format(e))
                    logger.state_update(epoch=e, 
                                        valid_loss=loss.item(), 
                                        model=self.model,
                                        optimizer=optimizer)
                    logger.save_ckp(False)
            
    def _standard_predict(self, dataloader):
        self.model.eval()
        test_set_size = dataloader.dataset.y_out.shape[0]
        sum_squre_error = torch.zeros(self.output_dim, dtype=torch.float)

        if self.cuda:
            sum_squre_error = sum_squre_error.cuda()
        for sample_batched in (dataloader):
            # taking the sample batch
            x_batch = sample_batched['Xsample']
            y_batch = sample_batched['Ysample']
            if self.cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            pred_batch = self.model(x_batch)
            sqr_batch = torch.sum(torch.square(y_batch - pred_batch), dim=0)
            sum_squre_error = sum_squre_error + sqr_batch
        sum_squre_error = sum_squre_error.cpu().data.numpy()
        nmse_standard_pred = sum_squre_error/test_set_size
        
        return nmse_standard_pred

    def _mcd_predict(self, dataloader, MCD_sample, unnormalized_output):
        self.model.train()
        start = time.time()
        pred_norm_list = []
        y_out = dataloader.dataset.y_out
        X_out = dataloader.dataset.X_out
        if unnormalized_output:
            y_out_mean = np.array(dataloader.dataset.y_trn_mean, ndmin=2)
            y_out_std = np.array(dataloader.dataset.y_trn_std, ndmin=2)
            def unnormalize(input_array):
                return y_out_mean + y_out_std*input_array
        X_out = torch.from_numpy(X_out)
        if self.cuda:
            X_out = X_out.cuda()
        pred_list = []
        for _ in range(MCD_sample):
            pred_tensor = self.model(X_out)
            pred_list.append(pred_tensor.cpu().data.numpy())
        mc_runtime = time.time() - start
        pred_mcd_arr = np.array(pred_list)
        std_mcd = np.std(pred_mcd_arr, 0)
        if unnormalized_output:
            pred_mcd_arr = unnormalize(pred_mcd_arr)
            y_out = unnormalize(y_out)
        pred_mcd = np.mean(pred_mcd_arr, 0)
        mse_mc = np.mean((y_out - pred_mcd)**2., 0)

        # We compute the test log-likelihood
        test_ll_mc = (logsumexp(-0.5 * self.tau * (y_out[np.newaxis, :]  - pred_mcd_arr)**2., 0) 
            - np.log(MCD_sample) - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll_mc = np.mean(test_ll_mc, 0)

        return mse_mc, std_mcd, test_ll_mc, mc_runtime

    def _ana_predict(self, dataloader, use_cov, unnormalized_output):
        if unnormalized_output:
            y_out_mean = np.array(dataloader.dataset.y_trn_mean, ndmin=2)
            y_out_std = np.array(dataloader.dataset.y_trn_std, ndmin=2)
            def unnormalize(input_array):
                return y_out_mean + y_out_std*input_array
        # get model which also outputs the activations before dropout layer and deactivates dropout
        torch.cuda.empty_cache()
        approximator = UncertaintyPropagator(self.model, use_cov=use_cov, cuda=self.cuda)

        start = time.time()
        # switch to eval() becuase in train() the output of dropout layer is scaled up by 1/(1-p)
        approximator.unc_model.eval() 
        # approximator.model.eval()
        pred_tensor = None
        var_tensor = None
        for sample_batched in (dataloader):
            # taking the sample batch
            x_batch = sample_batched['Xsample']
            if self.cuda:
                x_batch = x_batch.cuda()

            pred_batch, var_batch, _ = approximator.predict_avp(x_batch)
            if pred_tensor is None:
                pred_tensor = pred_batch
            else:
                pred_tensor = torch.cat((pred_tensor, pred_batch), 0)
            
            if var_tensor is None:
                var_tensor = var_batch
            else:
                var_tensor = torch.cat((var_tensor, var_batch), 0)

        ana_runtime = time.time() - start
        # work in numpy
        if var_tensor is not None:
            std_ana = np.sqrt(var_tensor.cpu().data.numpy())
        else:
            out_shape = list(dataloader.dataset.y_out.shape)
            out_shape.append(out_shape[-1])
            std_ana = np.zeros(out_shape)
        pred_ana = pred_tensor.cpu().data.numpy()
        
        if use_cov:
            if std_ana.shape[-1] != 1:
                std_ana = std_ana.diagonal(0, -1, -2)
            else:
                std_ana = std_ana[:, :, 0]
        
        # two ways of determining the test log-likelihood
        # sample from distribution to compute test log-likelihood
        y_out = dataloader.dataset.y_out
        T_our = 1000
        samples_norm = np.array([np.random.normal(loc=pred_ana, scale=std_ana) for i in range(T_our)])
        if unnormalized_output:
            y_out = unnormalize(y_out)
            pred_ana = unnormalize(pred_ana)
            samples = unnormalize(samples_norm)
        else:
            samples = samples_norm

        test_ll_ana = (logsumexp(-0.5 * self.tau * (y_out[np.newaxis, :] - samples)**2., 0)
             - np.log(T_our) - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll_ana = np.mean(test_ll_ana, 0)

        mse_ana = np.mean((y_out - pred_ana) ** 2., 0)

        return mse_ana, std_ana, test_ll_ana, ana_runtime

    def predict(self, 
                dataset, 
                batch_size=512, 
                use_cov=False, 
                MCD_sample=0, 
                unnormalized_output=True,
                load_model=False,
                ckp_dir=""):
        """
            Function for making predictions with the Bayesian neural network.
            @param X_test  
            @param y_test    
            @param use_cov: flag for using covariance or variance
            @param MCD_sample  : number of samples for MCdropout
        """
        if load_model:
            logger = SaveAndLoad(checkpth=ckp_dir)
            checkpoint = logger.load_ckp(is_best=True)
            self.model.load_state_dict(checkpoint['state_dict'])

        num_workers = 0 if self.dataset.cuda else 8
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                num_workers=num_workers)
        
        # We compute the predictive mean and variance for the target variables
        # of the test data
        nmse_standard_pred = self._standard_predict(dataloader)
        if MCD_sample > 0: # do MC-dropout 
            mse_mc, std_mcd, test_ll_mc, mc_runtime = self._mcd_predict(dataloader,
                                                                        MCD_sample,
                                                                        unnormalized_output)
        mse_ana, std_ana, test_ll_ana, ana_runtime = self._ana_predict(dataloader,
                                                                        use_cov,
                                                                        unnormalized_output)

        if MCD_sample > 0:
            print("np.abs(std_mcd - std_ana).mean(): \033[91m {:.4f} \033[0m"
                .format(np.abs(std_mcd - std_ana).mean()))
            print('Mean std difference: \033[91m {:.4f} \033[0m'
                .format(np.abs(std_mcd - std_ana).mean() / std_mcd.mean()))

        # We are done!
        if MCD_sample > 0:
            return nmse_standard_pred, mse_mc, test_ll_mc, mse_ana, test_ll_ana, mc_runtime, ana_runtime
        else:
            return nmse_standard_pred, mse_ana, test_ll_ana, ana_runtime

