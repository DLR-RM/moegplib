""" This files contain clustering method that works on the latent variable itself.
"""
import scipy
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from moegplib.moegp.kernels import rbfkernel, kernel, ard_kernel


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        ''' initializes the methods
        args:
            n_clusters: number of clusters (scalar)
            max_iter: maximum number of iterations (scalar)
            tol: tolerance parameter (scalar)
            random_state:
            kernel: input to scipy pairwise_kernels
            gamma: 
            degree:
            coef0:
            kernel_params:
            verbose: boolean whether to say much.
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)
    
    
class KernelKMeansCustom(BaseEstimator, ClusterMixin):
    """
    Kernel K-means Custom.
    
    We modify the implementation of Mathieu Blondel to use the custom kernels
    that are not provided by pairwise_kernels from sklearn.
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="rbf", l_opt=0.1, sigma_f_opt=0.5, verbose=0):
        ''' initializes the methods
        args:
            n_clusters: number of clusters (scalar)
            max_iter: maximum number of iterations (scalar)
            tol: tolerance parameter (scalar)
            random_state:
            kernel: input to scipy pairwise_kernels
            l_opt: scalar kernel param
            sigma_f_opt: scalar kernel param
            verbose: boolean whether to say much.
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.l_opt = l_opt
        self.sigma_f_opt = sigma_f_opt
        self.verbose = verbose
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if self.kernel == 'rbf':
            K = rbfkernel(X, Y, l=self.l_opt, sigma_f=self.sigma_f_opt) 
        elif self.kernel == 'ard':
            K = ard_kernel(X, Y, l=self.l_opt, sigma_f=self.sigma_f_opt)
        else:
            print("unsupported kernel type")
            exit(0)
        return K

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]
        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)
    
    
def two_step_kernel_kmeans(X, m, n_clusters=5, max_iter=100, random_state=0, kernel="rbf", verbose=1):
    '''
    This functions computes the 2-step kernel Kmeans.
    Args:
        X: training data-set. n x d numpy array
        m: m subset pseudo points.
    Returns:
        cl: n, numpy array
        cl_sub: m, numpy array (note that indices are random ; so might ignore this)
    '''
    # subsampling the clustering data points
    r_ind = np.random.choice(X.shape[0], m, replace=False)  
    Xsub = X[r_ind]
    
    # init kernel kmeans
    km = KernelKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, kernel=kernel, verbose=verbose)
    
    # fit the points 
    cl_sub = km.fit_predict(Xsub)
    
    # clustering all the training data-points.
    cl = km.predict(X)
    
    # get all the cluster centers
    centroids = []
    for i in range(n_clusters):
        idx_e = np.nonzero(cl_sub==i)[0]
        centroids.append(np.mean(Xsub[idx_e], axis=0))
    
    return cl, cl_sub, centroids
