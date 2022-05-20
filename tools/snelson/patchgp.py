# -*- coding: utf-8 -*-
""" PatchGP on Snelson
"""
import argparse
import numpy as np
import torch
import gpytorch
import itertools

from moegplib.datasets.toydata import SnelsonDataset
from moegplib.utils.logger import SaveAndLoad, BoardLogger
from moegplib.networks.toydata import SnelsonPrimeNet
from moegplib.networks.modelquantiles import ModelQuantiles
from moegplib.clustering.quantiles import quantiles1d
from moegplib.moegp.gpytorch import NTK1DGP

import matplotlib.pyplot as plt


CUDA_LAUNCH_BLOCKING="1"


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="An example with Snelson data-set.")
    
    parser.add_argument('--seed', default=1234, type=int,
                        help='seed for numpy and pytorch (default: 1234)')
    parser.add_argument('--ckp_dir', default="./",
                        help='directory of the check point file.')
    parser.add_argument('--data_dir', default="./",
                        help='directory of the data')
    return parser.parse_args(args) 


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    # arguments
    args = parse_args(args)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("available devides:", device)

    # initialization
    lr = 0.1
    delta = 1.5
    cluster_nr = 7

    # create the dataset and data loader
    dataset = SnelsonDataset(data_dir = args.data_dir, inbetween=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True, num_workers=4)

    # create the model
    model = SnelsonPrimeNet(D_in=1, H=32, D_out=1, n_layers=2, activation='sigmoid')
    logger = SaveAndLoad(checkpth=args.ckp_dir)
    checkpoint = logger.load_ckp(is_best=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # output the datasets and clustering
    xtrain = dataloader.dataset.X
    ytrain = dataloader.dataset.Y
    xtest = dataloader.dataset.Xtest
    idx, idx_t, lidx, lidx_t, bnd_x, bidx, nb, CONN = quantiles1d(xtrain, xtest, cluster_nr)
    xtrain = xtrain[lidx].squeeze(-1)
    ytrain = ytrain[lidx].squeeze(-1)
    xtest = xtest[lidx_t].squeeze(-1)

    # put the data into a gpu.
    xtrain = torch.from_numpy(xtrain).to(device)
    ytrain = torch.from_numpy(ytrain).unsqueeze(-1).to(device)
    xtest = torch.from_numpy(xtest).to(device)
    bnd_x = torch.from_numpy(bnd_x).unsqueeze(-1).to(device)

    # compute the Jacobians with a gpu incl. boundary points (note: the input shapes should match)
    mq = ModelQuantiles(model=model, data=(xtrain.unsqueeze(-1), ytrain.unsqueeze(-1)), delta=delta, devices=device)
    (Jtrain, yhat, s_noise, m_0, S_0) = mq.projection()
    (Jtest, _, _, _, _) = mq.projection(xtest.unsqueeze(-1))
    (Jbnd, _, _, _, _) = mq.projection(bnd_x.unsqueeze(-1))

    dnnpred = list()
    gpsig2 = list()
    xtestlist = list()

    for i in range(len(idx)):
        # per experts, extract neighbors
        experts = list()

        # list neighborhoods (only valid for 2-D case)
        if not len(CONN):
            lst = list()
        else:
            lst_a = list() if np.nonzero(CONN[0] == i)[0].size == 0 else (np.nonzero(CONN[0] == i)[0]).astype(int)
            lst_b = list() if np.nonzero(CONN[1] == i)[0].size == 0 else (np.nonzero(CONN[1] == i)[0]).astype(int)
            if len(lst_a) and len(lst_b):
                lst = [lst_a, lst_b] # note this is only for 2-D case!
            elif len(lst_a) and not len(lst_b):
                lst = [lst_a]
            elif not len(lst_a) and len(lst_b):
                lst = [lst_b]
            else:
                print("no neighbor! - is it even possible?")
                lst = list()
            lst = sorted(list(itertools.chain(*lst)))
            experts.append([i])
            experts.append(CONN[1][lst_a].tolist())
            experts.append(CONN[0][lst_b].tolist())
            experts = list(itertools.chain(*experts))
            experts = [x+1 for x in experts]
            experts = list(filter(None, experts))
            experts = list(set(experts))
            experts = [x-1 for x in experts]
            experts = sorted(experts)
            bidx_e = list(np.asarray(bidx)[lst])
            
        # cut the matrices based on neighbors
        Xtrainhat = Jtrain[0][np.concatenate([idx[ei] for ei in experts])]
        Ytrainhat = yhat[0][np.concatenate([idx[ei] for ei in experts])]
        Xtesthat = Jtest[0][np.concatenate([idx_t[ei] for ei in experts])]
        
        # patchwork gp build train covariance VIA initiate likelihood and model.
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gpmodel = NTK1DGP(Xtrainhat, Ytrainhat, likelihood).to(device)

        # a new list comprehension
        idx_e = list()
        idx_t_e = list()
        cnt = 0
        cnt_t = 0
        lidx_e = np.arange(0, len([np.concatenate([idx[ei] for ei in experts])][0]))
        lidx_t_e = np.arange(0, len([np.concatenate([idx_t[ei] for ei in experts])][0]))
        for ei in experts:
            idx_e.append(np.arange(cnt, cnt+len(idx[ei])))
            idx_t_e.append(np.arange(cnt_t, cnt_t+len(idx_t[ei])))
            if ei == i:
                idx_ce = np.arange(cnt_t, cnt_t+len(idx_t[ei]), 1)
            cnt = cnt + len(idx[ei])
            cnt_t = cnt_t + len(idx_t[ei])
        bidxe_e = list(np.arange(0, len(bidx_e)))
        CONN_e = [CONN[0][bidx_e], CONN[1][bidx_e]]
        minus = CONN_e[0][0]
        CONN_e[0] = CONN_e[0] - minus
        CONN_e[1] = CONN_e[1] - minus     

        # training a gp model
        training_iter = 100
        gpmodel.train()
        likelihood.train()
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpmodel)

        # entering training loop
        for i in range(training_iter):
            optimizer.zero_grad()
            output = gpmodel(Xtrainhat)
            loss = -mll(output, Ytrainhat)
            loss.backward(retain_graph=True)
            if i%5==0:
                print('Iter %d/%d - Loss: %.3f   delta: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                gpmodel.covar_module.variance.item(),
                gpmodel.likelihood.noise.item()
                ))
            optimizer.step()
        
        # Get into evaluation (predictive posterior) mode
        #patchkwargs['is_train'] = False
        gpmodel.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(gpmodel(Xtesthat))
            sig2 = observed_pred.stddev.mul_(2)
            dnn_pred = model(xtest[np.concatenate([idx_t[ei] for ei in experts])]) # train statistics
            
            # loading the variables
            dnnpred.append(dnn_pred.cpu().numpy()[idx_ce])
            gpsig2.append(sig2.cpu().numpy()[idx_ce])

    dnnpred = np.concatenate(dnnpred)
    gpsig2 =  np.concatenate(gpsig2)

    figure, axis = plt.subplots(figsize=(4, 4))
    x=xtest.cpu().numpy().ravel()
    mean=dnnpred.ravel()
    e=gpsig2**0.5 # to standard deviation
    axis.fill_between(x, mean - e, mean + e, color='blue', alpha=0.2)
    axis.fill_between(x, mean - 2 * e, mean + 2 * e, color='blue', alpha=0.15)
    axis.fill_between(x, mean - 3 * e, mean + 3 * e, color='blue', alpha=0.15)
    axis.plot(x, mean, color='red')
    axis.scatter(xtrain.cpu().numpy().ravel(), ytrain.cpu().numpy().ravel(), color='black',s=5)
    axis.set_ylabel('y', fontsize=12)
    axis.set_xlabel('x', fontsize=12)
    plt.title("Ours")
    plt.tight_layout()
    plt.show()

def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
