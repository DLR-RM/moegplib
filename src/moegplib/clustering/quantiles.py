""" This files contain clustering method that works on the latent variable itself.
"""
import numpy as np


def quantiles1d(xtrain, xtest, M):
    # initialization
    n = np.shape(xtrain)[0] # number of test points
    bv = np.quantile(xtrain, np.arange(0, 1.0+1/M, 1/M))
    bv[0] = bv[0]-0.1 
    bv[M] = bv[M]+0.1
    cnt = 0
    cnt_t = 0
    lidx = np.zeros((n, 1), dtype=int)
    lidx_t = np.zeros((np.shape(xtest)[0], 1), dtype=int)
    idx = []
    idx_t = []
    bidx = []
    for k in range(M):
        idx_temp = np.nonzero((xtrain >= bv[k]) & (xtrain < bv[k+1]))[0]
        lidx[cnt:cnt+np.shape(idx_temp)[0], 0] = idx_temp
        idx_temp = np.arange(cnt,(cnt+np.shape(idx_temp)[0]), 1)
        cnt = cnt + np.shape(idx_temp)[0]
        idx.append(idx_temp)
        if k == 0:
            idx_t_temp = np.nonzero((xtest <= bv[k+1]))[0] 
        elif k == M-1:
            idx_t_temp = np.nonzero((xtest >= bv[k]))[0]
        else:
            idx_t_temp = np.nonzero((xtest >= bv[k]) & (xtest < bv[k+1]))[0] 
        lidx_t[cnt_t:cnt_t + np.shape(idx_t_temp)[0], 0] = idx_t_temp
        idx_t_temp = np.arange(cnt_t,(cnt_t+np.shape(idx_t_temp)[0]), 1)
        cnt_t = cnt_t + np.shape(idx_t_temp)[0]
        idx_t.append(idx_t_temp)
        if k < (M-1):
            bidx.append(k)
    bnd_x = bv[1:len(bv)-1] 
    nb = M-1
    CONN = [np.arange(0,M-1).T, np.arange(1,M).T]
    return idx, idx_t, lidx, lidx_t, bnd_x, bidx, nb, CONN