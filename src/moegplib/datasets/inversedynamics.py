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


class SarcosDataset(Dataset):
    """SARCOS dataset (w/o split)."""

    def __init__(self, data_dir, normalize=True, transform=None):
        """[initialization]

        Args:
            data_dir ([type]): [description]
            normalize (bool, optional): [description]. Defaults to True.
            transform ([type], optional): [description]. Defaults to None.
        """
        # define paths
        name_file = "sarcos_inv"
        name_file_test = "sarcos_inv_test"
        mat_file = data_dir + name_file + ".mat"
        mat_file_test = data_dir + name_file_test + ".mat"
        
        # read the data
        sarcos_inv = scipy.io.loadmat(mat_file)
        sarcos_inv_test = scipy.io.loadmat(mat_file_test)
        
        # assign the data variables
        self.X = sarcos_inv[name_file][:, :21]
        self.Y = sarcos_inv[name_file][:, 21:]
        self.Xtest = sarcos_inv_test[name_file_test][:, :21]
        self.Ytest = sarcos_inv_test[name_file_test][:, 21:]
        
        # normalize values
        self.std_x_train = np.std(self.X, 0)
        self.std_x_train[self.std_x_train == 0] = 1
        self.mean_x_train = np.mean(self.X, 0)
        self.std_y_train = np.std(self.Y, 0)
        self.std_y_train[self.std_y_train == 0] = 1
        self.mean_y_train = np.mean(self.Y, 0)
        
        if normalize:
            self.X = (self.X - np.full(self.X.shape, self.mean_x_train)) / \
                      np.full(self.X.shape, self.std_x_train)
            self.Xtest = (self.Xtest - np.full(self.Xtest.shape, self.mean_x_train)) / \
                          np.full(self.Xtest.shape, self.std_x_train)
            self.Y = (self.Y - np.full(self.Y.shape, self.mean_y_train)) / \
                      np.full(self.Y.shape, self.std_y_train)
            self.Ytest = (self.Ytest - np.full(self.Ytest.shape, self.mean_y_train)) / \
                          np.full(self.Ytest.shape, self.std_y_train)
        
        # define transforms and data_dir
        self.transform = transform
        self.data_dir = data_dir

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
    

class SarcosDatasetSplits(Dataset):
    """SARCOS dataset (including splits to make the results reproducible)."""

    def __init__(self, data_dir, normalize=True, transform=None, cuda=True, normalize_test_set=False):
        """ Different from SarcosDataset (includes splits)
        Args:
            data_dir (string): Directory with the mat files.
            transform (callable, optional): Optional transform to be applied on a sample.data_dir (string): 
            normalize (bool): whether to normalize training set and validation set
            normalize_test_set (bool): whether to normalize test set
            cuda (bool): whether to put the training data on GPU before training, which can
                        accerlerate the training
        """
        self.cuda = cuda
        
        # define paths
        name_file = "sarcos_inv"
        name_file_test = "sarcos_inv_test"
        mat_file = data_dir + name_file + ".mat"
        mat_file_test = data_dir + name_file_test + ".mat"
        
        # read the data
        sarcos_inv = scipy.io.loadmat(mat_file)
        sarcos_inv_test = scipy.io.loadmat(mat_file_test)
        
        # assign the data variables
        self.X_train_val = sarcos_inv[name_file][:, :21]
        self.y_train_val = sarcos_inv[name_file][:, 21:]
        self.X_test_ori = sarcos_inv_test[name_file_test][:, :21]
        self.y_test_ori = sarcos_inv_test[name_file_test][:, 21:]
        
        # define transforms and data_dir
        self.transform = transform
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalize_test_set = normalize_test_set

    def _normalize(self):
        """[summary]

        Raises:
            Exception: [description]
        """
        # obtain statistics
        print("Nomrlizing the data (normalize_test_set is {}):".format(self.normalize_test_set))
        self.X_trn_std = np.std(self.X_train, 0)
        self.X_trn_std[self.X_trn_std == 0] = 1
        self.X_trn_mean = np.mean(self.X_train, 0)
        self.y_trn_std = np.std(self.y_train, 0)
        self.y_trn_std[self.y_trn_std == 0] = 1
        self.y_trn_mean = np.mean(self.y_train, 0)
        
        # shape of the data
        print ('Shape of X_trn_mean examples: ' + str(self.X_trn_mean.shape))
        print ('Shape of X_trn_std examples: ' + str(self.X_trn_std.shape))
        print ('Shape of y_trn_mean examples: ' + str(self.y_trn_mean.shape))
        print ('Shape of y_trn_std examples: ' + str(self.y_trn_std.shape))

        # normalize train set
        self.X_train = (self.X_train - np.full(self.X_train.shape, self.X_trn_mean)) / \
        np.full(self.X_train.shape, self.X_trn_std)
        self.y_train = (self.y_train - np.full(self.y_train.shape, self.y_trn_mean)) / \
        np.full(self.y_train.shape, self.y_trn_std)
        
        # normalizing test set
        if self.normalize_test_set:
            self.X_test = (self.X_test - np.full(self.X_test.shape, self.X_trn_mean)) / \
            np.full(self.X_test.shape, self.X_trn_std)
            self.y_test = (self.y_test - np.full(self.y_test.shape, self.y_trn_mean)) / \
            np.full(self.y_test.shape, self.y_trn_std)
            
        # normalize validation set
        if self.split_trn_val is not None:
            if self.split_trn_val < 1.:
                self.X_validation = (self.X_validation - np.full(self.X_validation.shape, self.X_trn_mean)) / \
                np.full(self.X_validation.shape, self.X_trn_std)
                self.y_validation = (self.y_validation - np.full(self.y_validation.shape, self.y_trn_mean)) / \
                np.full(self.y_validation.shape, self.y_trn_std)
        else:
            raise Exception("self.split_trn_val is {}, \
                self.set_training_set() should be called!".format(self.split_trn_val))

    def set_training_split(self, split_trn_val=None):
        print("Setting training_split, split_trn_val is \033[91m{}\033[0m".format(split_trn_val))
        self.split_trn_val = split_trn_val
        
        if split_trn_val < 1.0:
            num_training_examples = int(split_trn_val* self.X_train_val.shape[0])
            self.X_train = copy.deepcopy(self.X_train_val[0:num_training_examples, :])
            self.y_train = copy.deepcopy(self.y_train_val[0:num_training_examples])
            self.X_validation = copy.deepcopy(self.X_train_val[num_training_examples:, :])
            self.y_validation = copy.deepcopy(self.y_train_val[num_training_examples:])
            # Printing the size of the training, validation and test sets
        elif split_trn_val == 1.0:
            self.X_train = copy.deepcopy(self.X_train_val)
            self.y_train = copy.deepcopy(self.y_train_val)
            self.X_validation = np.array([])
            self.y_validation = np.array([])
        self.X_test = copy.deepcopy(self.X_test_ori)
        self.y_test = copy.deepcopy(self.y_test_ori)
        if self.normalize:
            self._normalize()
        if self.cuda:
            self.X_train = torch.from_numpy(self.X_train).cuda()
            self.y_train = torch.from_numpy(self.y_train).cuda()
        self.X_out = self.X_train
        self.y_out = self.y_train

    def choose_output_set(self, output_set="train"):
        if output_set == "test":
            self.X_out = self.X_test
            self.y_out = self.y_test
        elif output_set == "train":
            self.X_out = self.X_train
            self.y_out = self.y_train
        elif output_set == "validation":
            if self.split_trn_val < 1.0:
                self.X_out = self.X_validation
                self.y_out = self.y_validation
            else:
                raise Exception("split_trn_val = 1.0, no validation set!")
        else:
            raise Exception("Wrong argument: ", output_set)

    def __len__(self):
        return len(self.X_out)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if self.cuda:
                idx.cuda()
        Xsample = self.X_out[idx, :]
        Ysample = self.y_out[idx, :]
        sample = {'Xsample': Xsample, 'Ysample': Ysample}
        if self.transform:
            sample = self.transform(sample)
        return sample


class KukaDataset(Dataset):
    """Kuka dataset."""

    def __init__(self, data_dir, normalize=True, transform=None, cuda=True, normalize_test_set=False, sim_flag=False):
        """
        Args:
            data_dir (string): Directory with the txt files.
            transform (callable, optional): Optional transform to be applied on a sample.
            normalize (bool): whether to normalize training set and validation set
            normalize_test_set (bool): whether to normalize test set
            cuda (bool): whether to put the training data on GPU before training, which can accerlerate the training
            sim_flag(bool): whether the data is in simulation or in reality.
        """
        # init
        self.cuda = cuda
        
        # define paths
        trn_data_path = data_dir + "kuka_train.txt" 
        tst_data_path = data_dir + "kuka_test.txt"
        if sim_flag:
            trn_data_path = data_dir + "kuka_sim_train_"
            tst_data_path = data_dir + "kuka_sim_test.txt"

        # read the data
        if sim_flag:
            for part_idx in range(1, 5):
                part_trn_data_path = trn_data_path + "part{}.txt".format(part_idx)
                if part_idx == 1:
                    trn_data = np.loadtxt(part_trn_data_path)
                else:
                    trn_data = np.vstack((trn_data, np.loadtxt(part_trn_data_path)))
        else:
            trn_data = np.loadtxt(trn_data_path)
        tst_data = np.loadtxt(tst_data_path)
        
        # assign the data variables
        self.X_train_val = trn_data[:, :21]
        self.y_train_val = trn_data[:, 21:]
        self.X_test_ori = tst_data[:, :21]
        self.y_test_ori = tst_data[:, 21:]
        
        # this doesn't matter because it can be cancelled in the MSE
        self.y_trn_mean = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 
        self.y_trn_std = np.array([40.0, 40.0, 20.0, 20.0, 4.0, 4.0, 2.0])
        if sim_flag:
            self.y_trn_std = np.array([50.0, 70.0, 20.0, 20.0, 4, 4.0, 2.0])
        
        # define transforms and data_dir
        self.transform = transform
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalize_test_set = normalize_test_set

    def _normalize(self):
        # obtain statistics
        self.X_trn_std = np.std(self.X_train, 0)
        self.X_trn_std[self.X_trn_std == 0] = 1
        self.X_trn_mean = np.mean(self.X_train, 0)
        self.y_trn_std = np.std(self.y_train, 0)
        self.y_trn_std[self.y_trn_std == 0] = 1
        self.y_trn_mean = np.mean(self.y_train, 0)

        # normalize dataset
        self.X_train = (self.X_train - np.full(self.X_train.shape, self.X_trn_mean)) / \
        np.full(self.X_train.shape, self.X_trn_std)
        self.y_train = (self.y_train - np.full(self.y_train.shape, self.y_trn_mean)) / \
        np.full(self.y_train.shape, self.y_trn_std)
        if self.normalize_test_set:
            self.X_test = (self.X_test - np.full(self.X_test.shape, self.X_trn_mean)) / \
            np.full(self.X_test.shape, self.X_trn_std)
            self.y_test = (self.y_test - np.full(self.y_test.shape, self.y_trn_mean)) / \
            np.full(self.y_test.shape, self.y_trn_std)
        if self.split_trn_val is not None:
            if self.split_trn_val < 1.:
                self.X_validation = (self.X_validation - np.full(self.X_validation.shape, self.X_trn_mean)) / \
                np.full(self.X_validation.shape, self.X_trn_std)
                self.y_validation = (self.y_validation - np.full(self.y_validation.shape, self.y_trn_mean)) / \
                np.full(self.y_validation.shape, self.y_trn_std)
        else:
            raise Exception("self.split_trn_val is {}, \
                self.set_training_set() should be called!".format(self.split_trn_val))

    def set_training_split(self, split_trn_val=None):
        self.split_trn_val = split_trn_val
        if split_trn_val < 1.0:
            num_training_examples = int(split_trn_val* self.X_train_val.shape[0])
            self.X_train = copy.deepcopy(self.X_train_val[0:num_training_examples, :])
            self.y_train = copy.deepcopy(self.y_train_val[0:num_training_examples])
            self.X_validation = copy.deepcopy(self.X_train_val[num_training_examples:, :])
            self.y_validation = copy.deepcopy(self.y_train_val[num_training_examples:])
        elif split_trn_val == 1.0:
            self.X_train = copy.deepcopy(self.X_train_val)
            self.y_train = copy.deepcopy(self.y_train_val)
            self.X_validation = np.array([])
            self.y_validation = np.array([])
        self.X_test = copy.deepcopy(self.X_test_ori) 
        self.y_test = copy.deepcopy(self.y_test_ori) 
        if self.normalize:
            self._normalize()
        if self.cuda:
            self.X_train = torch.from_numpy(self.X_train).cuda()
            self.y_train = torch.from_numpy(self.y_train).cuda()
        self.X_out = self.X_train
        self.y_out = self.y_train

    def choose_output_set(self, output_set="train"):
        if output_set == "test":
            self.X_out = self.X_test
            self.y_out = self.y_test
        elif output_set == "train":
            self.X_out = self.X_train
            self.y_out = self.y_train
        elif output_set == "validation":
            if self.split_trn_val < 1.0:
                self.X_out = self.X_validation
                self.y_out = self.y_validation
            else:
                raise Exception("split_trn_val = 1.0, no validation set!")
        else:
            raise Exception("!!! Wrong argument: ", output_set)
        
    def __len__(self):
        return len(self.X_out)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if self.cuda:
                idx.cuda()
        Xsample = self.X_out[idx, :]
        Ysample = self.y_out[idx, :]
        sample = {'Xsample': Xsample, 'Ysample': Ysample}
        if self.transform:
            sample = self.transform(sample)
        return sample


def sarcos(root: str):
    """The SARCOS inverse kinematics dataset (https://github.com/Kaixhin/SARCOS).
    
    Args:
        root: Path to directory containing "sarcos_inv.mat" and "sarcos_inf_test.mat".

    Returns:
        The SARCOS train and test set as Numpy arrays.
    """
    sarcos_inv = scipy.io.loadmat(os.path.join(root, "sarcos_inv.mat"))
    sarcos_inv_test = scipy.io.loadmat(os.path.join(root, "sarcos_inv_test.mat"))
    x_train = sarcos_inv["sarcos_inv"][:, :21]
    y_train = sarcos_inv["sarcos_inv"][:, 21:]
    x_test = sarcos_inv_test["sarcos_inv_test"][:, :21]
    y_test = sarcos_inv_test["sarcos_inv_test"][:, 21:]
    return (x_train, y_train), (x_test, y_test)


def kuka(root: str, part: int = 1):
    """The KUKA inverse kinematics dataset (https://github.com/fmeier/kuka-data)

    Args:
        root: Path to directory containing "kuka1_online.txt" and "kuka1_offline.txt".
              Same for part 2.
        part: KUKA consists of two parts, 1 and 2. Select 0 for simulation data.

    Returns:
        The KUKA train and test set of the chosen dataset part as Numpy arrays.
    """
    if part > 0:
        train = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_online.txt"))
        test = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_offline.txt"))
    else:
        train = list()
        for p in [1, 2, 3, 4]:
            train.append(np.loadtxt(os.path.join(root, "kuka_sim_dataset", f"kuka_sim_train_part{p}.txt")))
        train = np.concatenate(train)
        test = np.loadtxt(os.path.join(root, "kuka_sim_dataset", f"kuka_sim_test.txt"))
    x_train = train[:, :21]
    y_train = train[:, 21:]
    x_test = test[:, :21]
    y_test = test[:, 21:]
    return (x_train, y_train), (x_test, y_test)