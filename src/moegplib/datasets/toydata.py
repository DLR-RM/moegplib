""" dataset.py contains scripts related to dataset loading.
"""

import os
import torch
import copy
import scipy.io
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from numpy import genfromtxt


class ToyData1D:
    def __init__(self, train_x, train_y, test_x=None, x_min=None, x_max=None, n_test=None, normalize=False, dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        if test_x is not None:
            self.test_x = np.array(test_x, dtype=dtype)[:, None]
            self.n_test = self.test_x.shape[0]
        else:
            self.n_test = n_test
            self.test_x = np.linspace(x_min, x_max, num=n_test, dtype=dtype)[:, None]


def load_snelson_data(n=200, dtype=np.float64):
    if n > 200:
        raise ValueError('Only 200 data points on snelson.')
    def _load_snelson(filename):
        with open('snelson/{fn}'.format(fn=filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)
    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)
   
    
class SnelsonDataset(Dataset):
    """Snelson dataset."""

    def __init__(self, data_dir='snelson', permute=True, inbetween=True, n=200, transform=None):
        """
        Args:
        """
        # define paths and load
        self.transform = transform
        self.data_dir = data_dir
        self.dtype = np.float64
        train_x = self._load_snelson("train_inputs")
        train_y = self._load_snelson("train_outputs")
        test_x = self._load_snelson("test_inputs")
        if n > 200:
            raise ValueError('Only 200 data points on snelson.')
        
        # permute and preprocess
        if permute:
            perm = np.random.permutation(train_x.shape[0])
            train_x = train_x[perm][:n]
            train_y = train_y[perm][:n]
        
        # assign the public variable 
        self.X = np.array(train_x, dtype=self.dtype)[:, None]
        self.Y = np.array(train_y, dtype=self.dtype)[:, None].reshape((-1,))  # snelson_data.train_y.reshape((-1,))
        self.n_train = self.X.shape[0]
        self.Xtest = np.array(test_x, dtype=self.dtype)[:, None]
        self.n_test = self.Xtest.shape[0]
        
        if inbetween:
            # making an in-between data
            X_test_snelson = np.linspace(-4, 10, 1000).reshape((-1, 1))
            
            # this is to evaluate in-between uncertainty
            mask = ((self.X < 1.5) | (self.X > 3)).flatten()
            self.X = self.X[mask, :]
            self.Y = self.Y[mask]
            self.Xtest = X_test_snelson
             
    def _load_snelson(self, filename):
        with open(self.data_dir +'snelson/{fn}'.format(fn=filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")], dtype=self.dtype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        Xsample = np.array([self.X[idx]])
        Ysample = np.array([self.Y[idx]])
        sample = {'Xsample': Xsample, 'Ysample': Ysample}
        if self.transform:
            sample = self.transform(sample)
        return sample