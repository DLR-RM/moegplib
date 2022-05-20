''' Metrics related functions and classes.
'''
import numpy as np
import copy
import torch
from scipy.special import logsumexp


def rmse_f(pred, ydata, unnormalize=False, ymean=None, ystd=None):
    if unnormalize:
        pred = ymean + ystd * pred
        ydata = ymean + ystd * ydata
    rmse = np.mean((ydata - pred) ** 2., 0) ** 0.5
    return rmse


def nll_f(pred, ydata, sig2, unnormalize=False, ymean=None, ystd=None):
    if unnormalize:
        pred = ymean + ystd * pred
        ydata = ymean + ystd * ydata
        std = ystd ** 2 * sig2
    test_ll = -0.5 * np.log(2 * np.pi * sig2) - 1.0 / (2.0 * sig2) * (ydata - pred) ** 2
    test_ll = np.mean(test_ll)
    return test_ll
