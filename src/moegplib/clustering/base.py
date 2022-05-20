""" This files contain clustering method that works on the latent variable itself.
"""
import pickle
import torch
import scipy
import logging
import os
import psutil
import operator
import concurrent.futures
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_kernels
from sklearn.decomposition import PCA, KernelPCA
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi, voronoi_plot_2d 
from scipy import linalg

from moegplib.networks.modelquantiles import ModelQuantiles
from moegplib.moegp.compression import JacobianPruner

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class boundary_data():
    """Data container class (use other type?)
    
    Defined at the top level to enable it as a picakble object.
    """
    def __init__(self, trainlabels, testlabels, xmn, kxmn, 
                    bidx, CONN, nb, idx, idx_t, lidx, lidx_t,
                    transformed_xsub=None, transformed_xtrain=None, 
                    transformed_xtest=None, is_visualize=False):
        self.trainlabels = trainlabels 
        self.testlabels = testlabels 
        self.xmn = xmn 
        self.kxmn = kxmn
        self.bidx = bidx 
        self.CONN = CONN 
        self.nb = nb
        self.idx = idx
        self.idx_t = idx_t
        self.lidx = lidx
        self.lidx_t = lidx_t
        if is_visualize:
            self.transformed_xsub = transformed_xsub
            self.transformed_xtrain = transformed_xtrain
            self.transformed_xtest = transformed_xtest


class DivisionStepBase:
    """ A parent class to data division.
    """
    def __init__(self, model, Jsaveload, args, n_princomp=2, 
                init_nr=100, max_iter=100, tol=1e-3, random_state=None, delta=1,
                targetout=0, init_method='k-means++', alpha=1, is_float=False,
                savermode="npy", is_zeroout=False, saver=None):
        """Initialization

        Args:
            model (torch.nn.Module): pytorch model based on nn.modules. 
            Jsaveload (object): a class that saves and loads the model quantiles
            args (object): parse arguments
            n_princomp (int, optional): number of principle components. Defaults to 2.
            n_clusters (int, optional): number of clusters. Defaults to 3.
            init_nr (int, optional): k-means initialization numbers. Defaults to 10.
            max_iter (int, optional): k-means maximum number of iterations. Defaults to 50.
            tol (float, optional): k-means tolerance. Defaults to 1e-3.
            random_state (float, optional): k-means random state. Defaults to None.
            delta (int, optional): prior precision parameter. Defaults to 1.
            init_method (str, optional): k-means initialization method. Defaults to 'k-means++'.
            alpha (int, optional): ridge regression parameter. Defaults to 1.
            targetout (int, optional): selects the output dimension. 
                                if dnn output is 7, 0 selects the first one. Defaults to 0.
        """
        self.args = args
        self.n_clusters = args.nr_experts
        self.m = args.nr_subset
        self.B = args.nr_boundaries
        self.n_princomp = n_princomp
        self.targetout = targetout
        self.is_float = is_float
        self.savermode = savermode
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_nr = init_nr
        self.init_method = init_method
        self.alpha = alpha
        self.delta = delta
        self.model = model
        self.Jsaveload = Jsaveload
        if args.pre_sparsity == 0.0:
            self.is_zeroout = False
        else:        
            self.is_zeroout = is_zeroout
        if self.is_zeroout:
            if saver is not None:
                masks_file = self.args.checkpoint_dir + '/' + saver + 'mask_pruned_dnn.pth'
            else:
                masks_file = self.args.checkpoint_dir + 'mask_pruned_dnn.pth'
            self.pruner = JacobianPruner(sparsity=None, masks_file=masks_file,
                                         mode='zeroesout')

    def divisionstep(self, dataset, testdataset):
        """Division step execution

        Args:
            dataset (torch dataloader): torch dataloader for training set
            testdataset (torch dataloader): torch dataloader for test set

        Returns:
            boundary_data: boundary_data class data container
        """
        # two step ntk pca
        logging.info("2-step kernel pca with ntk.")
        transformed_xsub, transformed_xtrain, \
            transformed_xtest = self._two_step_kernel_pca(dataset, testdataset)
        
        # clustering - fit and predict
        logging.info("Kernel-kmeans clustering.")
        sublabels, self.centroids \
            = self._kmeans_clustering(transformed_Xsub=transformed_xsub)

        logging.info("Kernel-kmeans prediction.")
        self.trainlabels = self._kmeans_prediction(Xtest=transformed_xtrain)
        print ("train cluster labels: ", self.trainlabels)
        self.testlabels = self._kmeans_prediction(Xtest=transformed_xtest)
        print ("train cluster labels: ", self.testlabels)

        # boundary generations
        logging.info("Extracting indices.")
        xmn, kxmn, bidx, CONN, nb = self._generate_boundary_points()
        idx, idx_t, lidx, lidx_t = self._extract_indices(transformed_xtrain,
                                                         transformed_xtest,
                                                         self.trainlabels,
                                                         self.testlabels,
                                                         self.n_clusters)

        self.bdata = boundary_data(self.trainlabels, self.testlabels,
                                   xmn, kxmn, bidx, CONN, nb,
                                   idx, idx_t, lidx, lidx_t,
                                   transformed_xsub, transformed_xtrain,
                                   transformed_xtest, False)
        return self.bdata

    def _ntk_two_step(self, dataset):
        """ Computes NTK in batch.

        Args:
            dataset (torch dataloader): pytorch dataloader (can be for both training and testing).
                                        this variable is passed to _jacobian_two_step.

        Returns:
            K (torch.Tensor): the NTK matrix of the subset. m x m matrix where m is a subset
                              chosen for scalable kernel PCA.
        """
        # Jacobian computations for the two step method
        self._jacobian_two_step(dataset)

        # initialize the target
        K = np.zeros((self.m, self.m))
        ntklow = 0

        # constructing the kernel matrix
        logging.info("Constructing the NTK matrix for subset m")

        for i in range(int(self._count)):
            # load the Jacobians and compute the row
            self.Jsaveload.mode = str(self.targetout) + "subset/"
            if self.max_batch_size == self.m:
                Jsub = self.Jsaveload.load_ckp(i)['Jtrain'] # TODO: make assertions
                K[:, :] = torch.cat([(1.0/self.delta) * \
                        Jsub @ self.Jsaveload.load_ckp(countnr)['Jtrain'].T \
                        for countnr in range(int(self._count))]).cpu().numpy()
            else:
                Jsub = self.Jsaveload.load_ckp(i)['Jtrain']
                Ktemp = torch.cat([(1.0/self.delta) * \
                        Jsub @ self.Jsaveload.load_ckp(countnr)['Jtrain'].T \
                        for countnr in range(int(self._count))], 1)

                K[ntklow:ntklow+Ktemp.shape[0], :] = Ktemp.cpu().numpy()

                ntklow = ntklow + Ktemp.shape[0]
        logging.info("the NTK matrix of the shape: %s by %s", str(K.shape[0]), str(K.shape[1]))
        return K

    def _two_step_kernel_pca(self, dataset, testdataset, train=True, test=True):
        """ Computes kernel pca on the subset of passed m data points,
        which are randomly selected.

        Args:
            dataset (torch dataloader): torch dataloader for training set
            testdataset (torch dataloader): torch dataloader for test set
            train (bool, optional): a boolean to enable train. Defaults to True.
            test (bool, optional): a boolean to enable test. Defaults to True.

        Returns:
            transformed_Xsub (torch.Tensor): PCA projected data for subset m.
            transformed_Xtrain (torch.Tensor): PCA projected data for training set.
            transformed_Xtest (torch.Tensor): PCA projected data for test set.
        """
        # kernel PCA with precomputed gram matrix.
        logging.info("Fit transform with kernel PCA")
        self.KPCA = KernelPCA(self.n_princomp, kernel='precomputed')
        transformed_Xsub = self.KPCA.fit_transform(self._ntk_two_step(dataset))

        # save the clustering model on ckp directory
        filename = self.args.checkpoint_dir + '/ntkpca' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        pickle.dump(self.KPCA, open(filename, 'wb'))

        # transform the train data
        if train:
            logging.info("NTK PCA with train data")
            transformed_Xtrain = self._ntk_pca(dataset, True)

        else:
            transformed_Xtrain = None

        # transform the test data
        if test:
            logging.info("NTK PCA with test data")
            transformed_Xtest = self._ntk_pca(testdataset, False)
        else:
            transformed_Xtest = None

        return transformed_Xsub, transformed_Xtrain, transformed_Xtest

    def _kmeans_clustering(self, transformed_Xsub=None):
        """ Computes kmeans clustering on the ntk pca

        Args: 
            transformed_Xsub (np.array, optional): Principal components of the training data. Defaults to None.

        Returns:
            [type]: [description]
        """
        # define K-means variable and do clustering
        k_means = KMeans(init=self.init_method, n_clusters=self.n_clusters, n_init=self.init_nr)
        k_means.fit(transformed_Xsub)

        # save the clustering model on ckp directory
        filename = self.args.checkpoint_dir + '/kmeans' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        pickle.dump(k_means, open(filename, 'wb'))

        # obtain the labels for each training data
        self.cluster_center = k_means.cluster_centers_
        trainlabels = pairwise_distances_argmin(transformed_Xsub, self.cluster_center)

        return trainlabels, self.cluster_center

    def _kmeans_prediction(self, Xtest=None):
        """ Computes cluster labels with kmeans for the new data.

        Args:
            Xtest (np.array, optional): array of principal components on test set. Defaults to None.

        Returns:
            testlabels (np.array): results of clustering on test set.
        """
        # loading the K-means clustering model FIXME checkpoint to args
        filename = self.args.checkpoint_dir + '/kmeans' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        loaded_model = pickle.load(open(filename, 'rb'))

        # obtain the labels for each training data
        self.k_means_cluster_centers = loaded_model.cluster_centers_

        # test labels
        testlabels = pairwise_distances_argmin(Xtest, self.k_means_cluster_centers)

        return testlabels

    def _extract_indices(self, Xtrain, Xtest, trainlabels, testlabels, n_clusters):
        """ Returns indices according to patchwork gp rules.
        The data type and their values are storngly required.

        Args:
            Xtrain (np.array): array of principal components on train set.
            Xtest (np.array): array of principal components on test set.
            trainlabels (np.array): results of clustering on training set.
            testlabels (np.array): results of clustering on test set.
            n_clusters (int): number of clusters.

        Returns:
            idx (list): list containing the expert indices for training.
            idx_t (list): list containing the expert indices for testing.
            lidx (list): list containing the ordering of idx for training.
            lidx_t (list): list containing the ordering of idx for testing.
        """
        # initialization
        idx, idx_t = list(), list()
        lidx = np.zeros((Xtrain.shape[0], 1), dtype=int)
        lidx_t = np.zeros((Xtest.shape[0], 1), dtype=int)
        cnt, cnt_t = 0, 0

        # clusters assigned with nr. experts
        for i in range(n_clusters):
            idx_temp = np.nonzero((trainlabels == i))[0] # left of the tree
            idx_t_temp = np.nonzero((testlabels == i))[0] # left of the tree
            idx.append(idx_temp)
            idx_t.append(idx_t_temp)

        # rescale and lidx computations
        for k in range(len(idx)):
            idx_temp = idx[k]
            lidx[cnt:cnt+idx_temp.shape[0], 0] = idx_temp
            idx[k] = np.arange(cnt,(cnt+np.shape(idx_temp)[0]), 1)
            cnt = cnt + np.shape(idx_temp)[0]
            idx_t_temp = idx_t[k]
            lidx_t[cnt_t:cnt_t+idx_t_temp.shape[0], 0] = idx_t_temp
            idx_t[k] = np.arange(cnt_t,(cnt_t+np.shape(idx_t_temp)[0]), 1)
            cnt_t = cnt_t + np.shape(idx_t_temp)[0]
        return idx, idx_t, lidx, lidx_t

    def _generate_boundary_points(self, B=None):
        """[Computes the boundary points from voroni method.
        Note that centroid or cluster centers have to be 2 dim.]

        Args:
            B ([type]): [description]
            plot (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # initialization
        xmn, kxmn, bidx, CONN = list(), list(), list(), list()
        R_b = 0 # 2nd boundary indice initialization

        # voronoi method
        vor = Voronoi(self.cluster_center)

        # boundary point extraction (ridge parts)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            CONN.append(np.asarray(pointidx))

        # now put them in the same data type!
        CONN = list(np.vstack(CONN).T)

        return xmn, kxmn, bidx, CONN, None


class DivisionStepNeuralTangent:
    """
    Division step using the neural tangent kernel.
    
    Reference
    ---------
    Should be written in the paper.
    """

    def __init__(self, model, Jsaveload, args, checkpoint=None, n_princomp=2, max_batch_size=500,
                init_nr=100, max_iter=100, tol=1e-3, random_state=None, delta=1,
                targetout=0, init_method='k-means++', alpha=1, is_float=False,
                savermode="npy", is_zeroout=True, saver=None, param_nr=126407):
        """[summary]

        Args:
            model ([type]): [description]
            Jsaveload ([type]): [description]
            args ([type]): [description]
            checkpoint ([type], optional): [description]. Defaults to None.
            n_princomp (int, optional): [number of principle components]. Defaults to 2.
            n_clusters (int, optional): [number of clusters]. Defaults to 3.
            init_nr (int, optional): [description]. Defaults to 10.
            m (int, optional): [number of pseudo points]. Defaults to 1000.
            B (int, optional): [number of boundary points]. Defaults to 50.
            max_iter (int, optional): [maximum number of iterations]. Defaults to 50.
            tol ([type], optional): [description]. Defaults to 1e-3.
            random_state ([type], optional): [description]. Defaults to None.
            delta (int, optional): [prior precision parameter]. Defaults to 1.
            init_method (str, optional): [k-means initialization method]. Defaults to 'k-means++'.
            alpha (int, optional): [ridge regression parameter]. Defaults to 1.
            targetout (int, optional): [selects the output dimension. 
                                if dnn output is 7, 0 selects the first one]. Defaults to 0.
        """
        self.args = args
        self.n_clusters = args.nr_experts
        self.m = args.nr_subset
        self.B = args.nr_boundaries
        self.n_princomp = n_princomp
        self.targetout = targetout
        self.is_float = is_float
        if args.pre_sparsity == 0.0:
            self.is_zeroout = False
        else:        
            self.is_zeroout = is_zeroout
        if self.is_zeroout:
            if saver is not None:
                masks_file = self.args.checkpoint_dir + '/' + saver + 'mask_pruned_dnn.pth'
            else:
                masks_file = self.args.checkpoint_dir + 'mask_pruned_dnn.pth'
            self.pruner = JacobianPruner(sparsity=None,
                                         masks_file=masks_file,
                                         mode='zeroesout')
        self.savermode = savermode
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.checkpoint = checkpoint
        self.init_nr = init_nr
        self.init_method = init_method
        self.alpha = alpha
        self.delta = delta
        self.model = model
        self.Jsaveload = Jsaveload
        self.max_batch_size = max_batch_size
        self.p = param_nr
        self._count = int(max(1, int(self.m / self.max_batch_size)))

    def divisionstep(self, dataset, testdataset, dev=None):
        """ Division step execution

        Args:
            dataset ([type]): [description]
            testdataset ([type]): [testdataset]
            dev ([torch.device]): if not None, the model will run on the specific device

        Returns:
            [type]: [description]
        """
        # two step ntk pca
        logging.info("2-step kernel pca with ntk.")
        transformed_xsub, transformed_xtrain, \
            transformed_xtest = self._two_step_kernel_pca(dataset, testdataset, dev=dev)
        
        # clustering - fit and predict
        logging.info("Kernel-kmeans clustering.")
        sublabels, self.centroids \
            = self._kmeans_clustering(transformed_Xsub=transformed_xsub)
            
        logging.info("Kernel-kmeans prediction.")
        if transformed_xtrain is not None:
            self.trainlabels = self._kmeans_prediction(Xtest=transformed_xtrain)
        else:
            self.trainlabels = None

        if transformed_xtest is not None:
            self.testlabels = self._kmeans_prediction(Xtest=transformed_xtest)
        else:
            self.testlabels = None
        
        # boundary generations
        logging.info("Generating boundary points and extracting indices.")
        xmn, kxmn, bidx, CONN, nb = self._generate_boundary_points(B=self.B)

        # extracting indices
        idx, idx_t, lidx, lidx_t = self._extract_indices(transformed_xtrain,
                                                         transformed_xtest,
                                                         self.trainlabels,
                                                         self.testlabels,
                                                         self.n_clusters)
                
        self.bdata = boundary_data(self.trainlabels, self.testlabels, 
                                   xmn, kxmn, bidx, CONN, nb, 
                                   idx, idx_t, lidx, lidx_t,
                                   transformed_xsub, transformed_xtrain,
                                   transformed_xtest, False)
        return self.bdata
    
    def expertsjacobiansaver(self, nr_expert, is_train=True):
        """Load and save Jacobians baesd on the batch algorithms.

        Args:
            bdata ([type]): [data class containing boundary points information]
            nr_expert (int): [the expert number specifically]

        Returns:
            [type]: [description]
        """
        # load the jacobians and find the data belonging to the expert
        n2m_lower, n2m_upper, Jind = 0, self.max_batch_size, 0 
        
        # function to get jacobians per expert
        def jacobian_per_experts(is_train, Jind, 
                                 n2m_lower, n2m_upper, 
                                 expert_nr, bdata):
            """ jacobians per expert
            Loads the previously saved jacobian,
            and selects the elements that belong to
            a specific expert.
            """       
            # initialization
            count = self._train_count
            idx = bdata.idx[expert_nr]
            if not is_train:
                idx = bdata.idx_t[expert_nr]
                count = self._test_count
            
            # data container
            self.Jsaveload.mode = "is_train" + str(is_train) + str(self.targetout)
            if self.is_zeroout:
                param_nr = self.p_pruned
            else:
                param_nr = self.p
            if self.savermode == 'zarr' or self.savermode == 'npy':
                Jx = np.zeros((len(idx), param_nr))
                Jy = np.zeros((len(idx)))
            elif self.savermode == 'cpk':
                Jx = torch.zeros([len(idx), param_nr], device='cpu')
                Jy = torch.zeros([len(idx)], device='cpu')
            else:
                raise AttributeError
            
            # per saved data with maximum batch size
            for j in range(count):
                logging.info("%s / %s Looping through saved jacobians @ expert %s :",
                                str(j), str(count-1), str(expert_nr))
                # loading data
                self.Jsaveload.mode = "is_train" + str(is_train) + str(self.targetout)
                if self.savermode == 'zarr' or self.savermode == 'npy':
                    loader = self.Jsaveload.load_numpy(j)
                elif self.savermode == 'cpk':
                    loader = self.Jsaveload.load_ckp(j)
                else:
                    raise AttributeError
                
                # ranges of indicies
                index = [x if x>=n2m_lower and x<n2m_upper else None for x in idx]
                index = [org for org in index if org is not None] 
                index = list(set(index))
                index = [int(x - j*self.max_batch_size) for x in index]
                
                # append data
                if index:
                    Jx[Jind:Jind+int(len(index)), :] \
                        = loader['Jtrain'][index, :]
                    Jy[Jind:Jind+int(len(index))] \
                        = loader['yhat'][index]
                
                # update the lower and upper bounds, and Jind
                n2m_lower += self.max_batch_size
                n2m_upper += self.max_batch_size
                Jind += len(index)
            
            return Jx, Jy

        # save the train jacobians per expert
        Jxtrain, Jytrain = jacobian_per_experts(is_train, Jind, 
                                                n2m_lower, n2m_upper, 
                                                nr_expert, self.bdata)
        self.Jsaveload.mode = "train_jacobians" + str(self.targetout)
        if self.savermode == 'zarr' or self.savermode == 'npy':
            self.Jsaveload.save_zarr(Jxtrain, None, 
                                        Jytrain, None,
                                    nrexpert=nr_expert, 
                                    is_verbose=False,
                                    mkdirs=False)
        elif self.savermode == 'cpk':
            self.Jsaveload.save_ckp(Jxtrain, None, 
                                    Jytrain, None,
                                    nrexpert=nr_expert)
            del Jxtrain, Jytrain
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        else:
            raise AttributeError

    def _jacobian_two_step(self, dataset, dev=None):
        """[Jacobian computations for dataloader in a batch]

        Args:
            datasets ([type]): [description]
            dev ([torch.device]): if not None, the model will run on the specific device
        """
        # initial parameters
        self._Xsub = list()
        
        # set the batch size
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=int(self.max_batch_size), 
                                                 shuffle=True, 
                                                 num_workers=4)
        repeat = max(1, int(self.m / self.max_batch_size))

        # the loop
        for batch_ndx, sample in enumerate(dataloader):
            logging.info("Looping through: %s", str(batch_ndx))
            
            # compute Jacobians
            if dev is not None:
                mq = ModelQuantiles(model=self.model.to(dev),
                                    data=(sample['Xsample'].to(dev), 
                                          sample['Ysample'].to(dev)), 
                                    delta=self.delta, 
                                    targetout=self.targetout,
                                    devices=dev)
                (Xhatsub, _, _, _, _) = mq.projection()
            else:
                mq = ModelQuantiles(model=self.model,
                                    data=(sample['Xsample'].to(self.args.device), 
                                          sample['Ysample'].to(self.args.device)), 
                                    delta=self.delta, 
                                    targetout=self.targetout,
                                    devices=self.args.device)
                (Xhatsub, _, _, _, _) = mq.projection()

            # skip the loop if Xhatsub is none
            if Xhatsub is None:
                continue
            
            # converting to a float
            if not self.is_float:
                Xhatsub = Xhatsub.half()
            
            # save the last jacobians
            if int((batch_ndx+1) * self.max_batch_size) >= int(self.m):
                logging.info("Saving the batch number: %s", str(batch_ndx))
                if batch_ndx > 0:
                    self.Jsaveload.mode = str(self.targetout) + "subset/"
                    cutind = int(self.max_batch_size) - int(int((batch_ndx+1) * self.max_batch_size) - int(self.m))
                    logging.info("Saving the batch number after cutting down: %s", str(cutind))
                    self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0)[0:cutind], 
                                            None, None, None, 
                                            nrexpert=batch_ndx)
                    self._Xsub.append(sample['Xsample'].cpu().numpy()[0:cutind])
                else:
                    self.Jsaveload.mode = str(self.targetout) + "subset/"
                    self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0), 
                                            None, None, None, 
                                            nrexpert=batch_ndx)
                    self._Xsub.append(sample['Xsample'].cpu().numpy())
                
                # stopping criteria for Jacobian compuations
                logging.info("Stopping: %s bigger than %s", 
                             str(int((batch_ndx+1) * self.max_batch_size)),
                             int(self.m))
                break
            
            # save all the jacobians
            if not int((batch_ndx+1) * self.max_batch_size) >= int(self.m):
                logging.info("Saving the batch number: %s", str(batch_ndx))
                self.Jsaveload.mode = str(self.targetout) + "subset/"
                self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0), 
                                        None, None, None, 
                                        nrexpert=batch_ndx)
                self._Xsub.append(sample['Xsample'].cpu().numpy())
    
    def _ntk_two_step(self, dataset, dev=None):
        """[Computes NTK in batch]

        Args:
            dataset ([type]): [description]
            dev ([torch.device]): if not None, the model will run on the specific device

        Returns:
            [type]: [description]
        """
        # Jacobian computations for the two step method
        self._jacobian_two_step(dataset, dev=dev)
         
        # initialize the target
        K = np.zeros((self.m, self.m))
        ntklow = 0
        
        # constructing the kernel matrix
        logging.info("Constructing the NTK matrix for subset m")
        for i in range(int(self._count)):
            # load the Jacobians and compute the row
            self.Jsaveload.mode = str(self.targetout) + "subset/"
            if self.max_batch_size == self.m:
                Jsub = self.Jsaveload.load_ckp(i)['Jtrain'] # TODO: make assertions 
                K[:, :] = torch.cat([(1.0/self.delta) * \
                        Jsub @ self.Jsaveload.load_ckp(countnr)['Jtrain'].T \
                        for countnr in range(int(self._count))]).cpu().numpy()
            else:
                Jsub = self.Jsaveload.load_ckp(i)['Jtrain']
                Ktemp = torch.cat([(1.0/self.delta) * \
                        Jsub @ self.Jsaveload.load_ckp(countnr)['Jtrain'].T \
                        for countnr in range(int(self._count))], 1)
                K[ntklow:ntklow+Ktemp.shape[0], :] = Ktemp.cpu().numpy()
                ntklow = ntklow + Ktemp.shape[0]
        logging.info("the NTK matrix of the shape: %s by %s", str(K.shape[0]), str(K.shape[1]))
        return K
        
    def _ntk_pca(self, dataset, is_train, dev=None):
        """Computes principle components of NTK.

        Args:
            dataloder ([type]): [description]
            dev ([torch.device]): if not None, the model will run on the specific device

        Returns:
            [type]: [description]
        """
        # Initialization
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=int(self.max_batch_size), 
                                                 shuffle=False, 
                                                 num_workers=4) # TODO: num_workers to change
        Xtransform = list()
        for batch_ndx, sample in enumerate(dataloader):
            logging.info("Looping through: %s", str(batch_ndx))
            
            # Jacobian computations
            if dev is not None:
                mq = ModelQuantiles(model=self.model.to(dev),
                                    data=(sample['Xsample'].to(dev),
                                          sample['Ysample'].to(dev)), 
                                    delta=self.delta, 
                                    targetout=self.targetout,
                                    devices=dev)
                (Xhat, yhat, _, _, _) = mq.projection()
            else:
                mq = ModelQuantiles(model=self.model,
                                    data=(sample['Xsample'].to(self.args.device),
                                          sample['Ysample'].to(self.args.device)), 
                                    delta=self.delta, 
                                    targetout=self.targetout,
                                    devices=self.args.device)
                (Xhat, yhat, _, _, _) = mq.projection()
            
            if not self.is_float:
                Xhat, yhat = Xhat.half(), yhat.half()
            
            # save the given variables
            self.Jsaveload.mode = "is_train" + str(is_train) + str(self.targetout)
            if self.savermode == 'zarr' or self.savermode == 'npy':
                if self.is_zeroout:
                    Xhatpruned = self.pruner.neuralprune(Xhat.squeeze(dim=0))
                    logging.info("Pruning results: from %s to %s", str(Xhat.shape), str(Xhatpruned.shape))
                    self.p_pruned = Xhatpruned.shape[1]
                    self.Jsaveload.save_numpy(Xhatpruned.cpu().detach().numpy(), None, 
                                              yhat.squeeze().cpu().detach().numpy(), None, 
                                              nrexpert=batch_ndx)
                else:
                    self.Jsaveload.save_numpy(Xhat.squeeze(dim=0).cpu().detach().numpy(), None, 
                                              yhat.squeeze().cpu().detach().numpy(), None, 
                                              nrexpert=batch_ndx)
            elif self.savermode == 'cpk':
                if self.is_zeroout:
                    Xhatpruned = self.pruner.neuralprune(Xhat.squeeze(dim=0))
                    logging.info("Pruning results: from %s to %s", str(Xhat.shape), str(Xhatpruned.shape))
                    self.p_pruned = Xhatpruned.shape[1]
                    self.Jsaveload.save_ckp(Xhatpruned.cpu(), None, 
                                            yhat.squeeze().cpu(), None, 
                                            nrexpert=batch_ndx)
                else:
                    self.Jsaveload.save_ckp(Xhat.squeeze(dim=0).cpu(), None, 
                                            yhat.squeeze().cpu(), None, 
                                            nrexpert=batch_ndx)
            else:
                raise AttributeError
            
            # applying pca and computing Xtransform
            self.Jsaveload.mode = str(self.targetout) + "subset/"
            NTK = torch.cat([(1.0/self.delta) * \
                    Xhat.squeeze(dim=0) @ self.Jsaveload.load_ckp(i)['Jtrain'].T \
                    for i in range(int(self._count))], 1).cpu().numpy()
            Xtransform.append(self.KPCA.transform(NTK))
            
            # delete variable
            del NTK, Xhat, yhat
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache() 
        
        # caching the batch
        if is_train:
            self._train_count = batch_ndx + 1
        else:
            self._test_count = batch_ndx + 1
        
        return np.concatenate(Xtransform)

    def _two_step_kernel_pca(self, dataset, testdataset, dev=None, train=True, test=True):
        """Computes kernel pca on the subset of passed m data points, 
        which are randomly selected.
        
        Args:
            dataloader ([type]): [description]
            testdataloader ([type]): [description]
            train (bool, optional): [description]. Defaults to True.
            test (bool, optional): [description]. Defaults to True.
            dev ([torch.device]): if not None, the model will run on the specific device

        Returns:
            [type]: [description]
        """
        # defining the filename
        filename = self.args.checkpoint_dir + '/ntkpca' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        
        if os.path.isfile(filename):
            # load the kpca object
            logging.info("Loading the NTK PCA")
            self.KPCA = pickle.load(open(filename, 'rb'))
            transformed_Xsub = None
        else:
            # kernel PCA with precomputed gram matrix.
            logging.info("Fit transform with kernel PCA")
            self.KPCA = KernelPCA(self.n_princomp, kernel='precomputed')
            transformed_Xsub = self.KPCA.fit_transform(self._ntk_two_step(dataset, dev=dev))

            # save the clustering model on ckp directory
            pickle.dump(self.KPCA, open(filename, 'wb'))

            # min max points to contrain boundary points.
            self.x1min = np.amin(transformed_Xsub[:,0])
            self.x1max = np.amax(transformed_Xsub[:,0])
            self.x2min = np.amin(transformed_Xsub[:,1])
            self.x2max = np.amax(transformed_Xsub[:,1])
            
        # transform the train data
        if dataset is not None:
            logging.info("NTK PCA with train data")
            transformed_Xtrain = self._ntk_pca(dataset, True, dev=dev) 
        else:
            transformed_Xtrain = None

        # transform the test data
        if testdataset is not None:
            logging.info("NTK PCA with test data")
            transformed_Xtest = self._ntk_pca(testdataset, False, dev=dev)
        else:
            transformed_Xtest = None

        return transformed_Xsub, transformed_Xtrain, transformed_Xtest
    
    def _kmeans_clustering(self, transformed_Xsub=None):
        """Computes kmeans clustering on the ntk pca.

        Args:
            transformed_Xsub ([type], optional): [principal components of the training data]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # defining the kmeans filename
        filename = self.checkpoint + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"

        if os.path.isfile(filename):
            k_means = pickle.load(open(filename, 'rb'))
        else:
            # define K-means variable and do clustering
            k_means = KMeans(init=self.init_method, n_clusters=self.n_clusters, n_init=self.init_nr)
            k_means.fit(transformed_Xsub)
            
            # save the clustering model on ckp directory
            pickle.dump(k_means, open(filename, 'wb'))
            
        # obtain the labels for each training data
        self.cluster_center = k_means.cluster_centers_

        # return the labels for the subset
        if transformed_Xsub is not None:
            trainlabels = pairwise_distances_argmin(transformed_Xsub, self.cluster_center)
        else:
            trainlabels = None
                    
        return trainlabels, self.cluster_center

    def _kmeans_prediction(self, Xtest=None):
        """Computes cluster labels with kmeans for the new data.

        Args:
            Xtest ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # loading the K-means clustering model
        filename = self.checkpoint + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        loaded_model = pickle.load(open(filename, 'rb'))

        # obtain the labels for each training data
        self.k_means_cluster_centers = loaded_model.cluster_centers_
        
        # test labels
        testlabels = pairwise_distances_argmin(Xtest, self.k_means_cluster_centers)
        
        return testlabels
    
    def _learn_pre_image_ntk(self):
        """[Computes the dual coefficient for the inverse transform.
        Acheived by using kernel ridge regression - hence learning the mapping.]
        
        ----- REFERENCE ------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        self.X4inv = self.KPCA.alphas_ * np.sqrt(self.KPCA.lambdas_) # short cut
        n_samples = self.X4inv.shape[0]
        K = (1/self.delta) * self.X4inv @ self.X4inv.T #self._ntk(Yhat=self.X4inv) # TODO: manually do it
        K.flat[::n_samples + 1] += self.alpha
        b = np.concatenate(self._Xsub)
        self.dual_coef = linalg.solve(K,
                                      b,
                                      sym_pos=True,
                                      overwrite_a=True)
    
    def _inverse_transform(self, X=None):
        """[Computes inverse of the kernel PCA]

        Args:
            X ([numpy array], (n_samples, n_components)): [description]. Defaults to None.

        Returns:
            [numpy array (n_samples, n_features)]: [description]
            
        ----- REFERENCE ------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        K = (1/self.delta) * X @ self.X4inv.T
        n_samples = self.X4inv.shape[0]
        K.flat[::n_samples + 1] += self.alpha
        return np.dot(K, self.dual_coef)
    
    def _sample_points(self, x, y, B):
        """ Computes points given two points in a line.
        
        Args:
            x ([type]): [description]
            y ([type]): [description]
            B ([type]): [description]

        Returns:
            [type]: [description]
        """
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 \
                   + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]
        fx, fy = interp1d(distance, x), interp1d(distance, y)
        beta = np.linspace(0, 1, B)
        x_regular, y_regular = fx(beta), fy(beta)
        return x_regular, y_regular

    def _generate_boundary_points(self, B=None):
        """ Computes the boundary points from voroni method.
        Note that centroid or cluster centers have to be 2 dim.

        Args:
            B ([type]): [description]
            plot (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # initialization
        xmn, kxmn, bidx, CONN = list(), list(), list(), list()
        R_b = 0 # 2nd boundary indice initialization 
        
        # voronoi method
        vor = Voronoi(self.cluster_center)
        
        # boundary point extraction (ridge parts)        
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            CONN.append(np.asarray(pointidx))
        
        # now put them in the same data type!
        CONN = list(np.vstack(CONN).T)
        
        return xmn, kxmn, bidx, CONN, None
    
    def _extract_indices(self, Xtrain, Xtest, trainlabels, testlabels, n_clusters):
        """[Returns indices according to patchwork gp rules.
        The data type and their values are storngly required.]

        Args:
            Xtrain ([type]): [description]
            Xtest ([type]): [description]
            trainlabels ([type]): [description]
            testlabels ([type]): [description]
            n_clusters ([type]): [description]

        Returns:
            [type]: [description]
        """
        if trainlabels is not None:
            # initialization
            idx, idx_t = list(), list()
            lidx = np.zeros((Xtrain.shape[0], 1), dtype=int)
            lidx_t = list()
            cnt, cnt_t = 0, 0

            # clusters assigned with nr. experts
            for i in range(n_clusters):
                idx_temp = np.nonzero((trainlabels == i))[0] # left of the tree
                idx.append(idx_temp)
                
            # rescale and lidx computations
            for k in range(len(idx)):
                idx_temp = idx[k]
                lidx[cnt:cnt+idx_temp.shape[0], 0] = idx_temp
                idx[k] = np.arange(cnt,(cnt+np.shape(idx_temp)[0]), 1)
                cnt = cnt + np.shape(idx_temp)[0]

        if testlabels is not None:
            # initialization
            idx, idx_t = list(), list()
            lidx = list()
            lidx_t = np.zeros((Xtest.shape[0], 1), dtype=int)
            cnt, cnt_t = 0, 0

            # clusters assigned with nr. experts
            for i in range(n_clusters):
                idx_t_temp = np.nonzero((testlabels == i))[0] # left of the tree
                idx_t.append(idx_t_temp)
                
            # rescale and lidx computations
            for k in range(len(idx_t)):
                idx_t_temp = idx_t[k]
                lidx_t[cnt_t:cnt_t+idx_t_temp.shape[0], 0] = idx_t_temp
                idx_t[k] = np.arange(cnt_t,(cnt_t+np.shape(idx_t_temp)[0]), 1)
                cnt_t = cnt_t + np.shape(idx_t_temp)[0]
            
        return idx, idx_t, lidx, lidx_t