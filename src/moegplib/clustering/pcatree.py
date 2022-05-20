""" This files contain a spatial clustering method.
"""
import torch
import scipy
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from moegplib.utils.utils import intersect_mtlb

   
class SpatialClustering():
    """
    Spatial tree based clustering [park et al 2018]
    """
    def __init__(self, xtrain, xtest, lv, K, device):
        self.idx, self.idx_t, self.lidx, self.lidx_t = self.pcatree_partition(xtrain, xtest, lv)
        xtrain = xtrain[self.lidx.ravel(), :]
        self.CONN, self.bidx, self.xmn, self.nb = self.gen_pseudodata(xtrain, self.idx, K)
        self.bnd_x = torch.from_numpy(self.xmn[self.lidx_t.ravel()]).to(device)

    def pcatree_partition(self, xtrain, xtest, lv):
        # simple introspection.
        if lv < 0: 
            print("K variable should be bigger than 2")
            exit(0)
            
        # init for the desired variables
        idx, idx_t = list(), list()
        lidx, lidx_t = np.zeros((xtrain.shape[0], 1), dtype=int), np.zeros((xtest.shape[0], 1), dtype=int)
        cnt, cnt_t = 0, 0
        region, region_t = np.zeros((xtrain.shape[0], 1)), np.zeros((xtest.shape[0], 1))
        pca = PCA(n_components=1)

        # loop through the tree from level 0 to level K
        for level in range(lv):
            # apply PCA and threshold
            if level == 0:
                pca.fit(xtrain)
                xtrain_red = pca.transform(xtrain) 
                threshold = np.median(xtrain_red)
                idx_l = np.nonzero((xtrain_red >= threshold))[0] 
                idx_r = np.nonzero((xtrain_red < threshold))[0] 
                idx.append(idx_l), idx.append(idx_r)
                xtest_red = pca.transform(xtest)
                idx_t_l = np.nonzero((xtest_red >= threshold))[0] 
                idx_t_r = np.nonzero((xtest_red < threshold))[0] 
                idx_t.append(idx_t_l), idx_t.append(idx_t_r)
            else:
                leaf_nr = len(idx)
                for ind in range(leaf_nr):
                    pca.fit(xtrain[idx[ind], :])
                    xtrain_red = pca.transform(xtrain[idx[ind], :])
                    threshold = np.median(xtrain_red)
                    idx_l = np.nonzero((xtrain_red >= threshold))[0] 
                    idx_r = np.nonzero((xtrain_red < threshold))[0]
                    idx.append(idx[ind][idx_l]), idx.append(idx[ind][idx_r])
                    xtest_red = pca.transform(xtest[idx_t[ind]])
                    idx_t_l = np.nonzero((xtest_red >= threshold))[0] 
                    idx_t_r = np.nonzero((xtest_red < threshold))[0] 
                    idx_t.append(idx_t[ind][idx_t_l]), idx_t.append(idx_t[ind][idx_t_r])
                idx, idx_t = idx[2**(level):], idx_t[2**(level):]
        for k in range(len(idx)):
            idx_temp = idx[k]
            lidx[cnt:cnt+idx_temp.shape[0], 0] = idx_temp
            idx[k] = np.arange(cnt,(cnt+np.shape(idx_temp)[0]), 1)
            region[idx_temp] = k
            cnt = cnt + np.shape(idx_temp)[0]
            idx_t_temp = idx_t[k]
            lidx_t[cnt_t:cnt_t+idx_t_temp.shape[0], 0] = idx_t_temp
            idx_t[k] = np.arange(cnt_t,(cnt_t+np.shape(idx_t_temp)[0]), 1)
            cnt_t = cnt_t + np.shape(idx_t_temp)[0]
        return idx, idx_t, lidx, lidx_t

    def gen_pseudodata(self, xtrain, idx, K):
        # knnsearch over K neighbors.
        neigh = NearestNeighbors(n_neighbors=K, metric="euclidean")
        neigh.fit(xtrain)
        neigh_dist, neigh_idx = neigh.kneighbors(xtrain)

        # generating boundary variables
        M = len(idx) 
        R_b = 0
        xmn = [] 
        CONN = [] 
        bidx = [] 
        for i in range(M):
            for j in range(i+1, M, 1):
                id_i = idx[i]
                id_j = idx[j]
                cij = neigh_idx[id_i, :]
                _, I, J = intersect_mtlb(cij, id_j)
                if len(I) != 0:
                    CONN.append(np.asarray([i, j]))
                    I, _ = np.unravel_index(I, cij.shape)
                    tmp = 0.5 * (xtrain[id_i[I], :] + xtrain[id_j[J], :])
                    bidx.append(np.arange(R_b, R_b+tmp.shape[0], 1))
                    xmn.append(tmp)
                    R_b = R_b + tmp.shape[0]

        # preprocessing some lists into numpy arrays & verification
        CONN = list(np.vstack(CONN).T)
        bidx = list(np.concatenate(bidx))
        xmn = np.concatenate(xmn) 
        nb = len(xmn) 
        return CONN, bidx, xmn, nb

    def delete_pseduodata(self):
        del self.xmn, self.bnd_x
        torch.cuda.empty_cache()