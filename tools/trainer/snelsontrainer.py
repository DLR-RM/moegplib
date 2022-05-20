# -*- coding: utf-8 -*-
""" Training script for the snelson data.
"""
import torch
import argparse
import sys
import numpy as np

from moegplib.datasets.toydata import SnelsonDataset
from moegplib.utils.logger import SaveAndLoad, BoardLogger
from moegplib.networks.toydata import SnelsonPrimeNet

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal, Bernoulli


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
    parser.add_argument('--ckp_dir', default="",
                        help='directory of the check point file.')
    parser.add_argument('--data_dir', default="",
                        help='directory of the data')
    return parser.parse_args(args) 


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    # arguments
    args = parse_args(args)
    
    # inititalization
    hidden_size = 32
    hidden_layers = 1
    input_size = 1
    output_size = 1
    activation = 'sigmoid'
    num_epoch = 10000  # the same as before  
    lr = 0.1  # 1e-3  # also called alpha
    lr_factor = 0.99  # todo: define this
    sigma_noise = 0.286
    delta = 0.1

    # create the dataset and data loader
    dataset = SnelsonDataset(data_dir=args.data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(),
                                             shuffle=True, num_workers=4)
    print("dataset size", dataset.__len__())
    
    # create the model
    model = SnelsonPrimeNet(D_in=input_size, H=hidden_size, D_out=output_size, n_layers=hidden_layers + 1, activation=activation)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    opt = Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(opt, 'min', factor=lr_factor, min_lr=1e-10)

    # define the check point class
    logger = SaveAndLoad(checkpth=args.ckp_dir)
    board = BoardLogger(checkpth=args.ckp_dir)

    # training init
    temploss = 10000 
    losses = list()

    # testing an iterator of the dataset
    for i in range(num_epoch):
        x_batch = torch.from_numpy(dataset.X).double()
        y_batch = torch.from_numpy(dataset.Y).double()
        opt.zero_grad()
        output = model.forward(x_batch)
        likelihood = Normal(output.flatten(), sigma_noise)
        prior = Normal(0, 1 / np.sqrt(delta))
        nll = - torch.sum(likelihood.log_prob(y_batch.double()))
        loss = nll - torch.sum(prior.log_prob(torch.cat([p.flatten() for p in model.parameters()])))
        loss.backward()
        opt.step()
        scheduler.step(loss.item())
        if (i+1) % 500 == 0:
            print(i + 1, nll.item())
        losses.append(loss.item())
            
        # Update and save the check point.
        logger.state_update(epoch=i, valid_loss=loss.item(), model=model, optimizer=opt)
        logger.save_ckp(False)
        if loss.item() < temploss:
            temploss = loss.item()
            logger.save_ckp(True)
            
        with torch.no_grad():
            # train statistics
            y_pred = model(torch.from_numpy(dataset.X).double())
            pred = y_pred.data.numpy()
            nMSE_train = sum((dataset.Y - pred.flatten())**2) / len(dataset.Y)
            
            # add tensorboard here
            board.lg_scalar("nmse_train_", nMSE_train, i)
            board.lg_scalar("total loss", loss.item(), i)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
