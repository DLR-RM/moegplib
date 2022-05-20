""" This files contain the division step for detr
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

from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset
from detectron2.data import DatasetMapper

from moegplib.networks.modelquantiles import DetectronQuantiles
from moegplib.moegp.compression import JacobianPruner
from moegplib.moegp.clustering.base import DivisionStepBase, boundary_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class DivisionStepDetectron(DivisionStepBase):
    def __init__(self, model, Jsaveload, args, saver, classout, regout, task, targetout, detr_cfg, is_zeroout=False):
        super().__init__(model=model, Jsaveload=Jsaveload, args=args, saver=saver, is_zeroout=is_zeroout)
        self.classout = classout
        self.regout = regout
        self.task = task
        self.targetout = targetout
        self.detr_cfg = detr_cfg
        self.p = self._compute_theta_map(model)
        self.pruneindex = torch.nonzero(self.p)[0]

    def expertsjacobiansaver(self, nr_expert):
        """ Load and save Jacobians baesd on the batch algorithms.
        After having computed the clusters and saved all the Jacobian matrices,
        this function loads the jacobian, picks the matrices that belong
        to a specific expert (given by nr_expert), and saves only the matrices
        that belong to this expert.

        Args:
            nr_expert (int): the expert number specifically, e.g. expert 1 or expert 200, etc.
        """
        # load the jacobians and find the data belonging to the expert
        n2m_lower, Jind, numcounter = 0, 0, 0

        # function to get jacobians per expert
        def jacobian_per_experts(is_train, Jind, numcounter,
                                 n2m_lower, expert_nr, bdata):
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
                index = [x if x>=n2m_lower and x<(n2m_lower+self._batchsizeslist[j]) else None for x in idx]
                index = [org for org in index if org is not None]
                index = list(set(index))
                index = [int(x - n2m_lower) for x in index]
                index = sorted(index) # sorting the index

                # append data
                if index:
                    Jx[Jind:Jind+int(len(index)), :] = loader['Jtrain'][index, :]
                    Jy[Jind:Jind+int(len(index))] = loader['yhat'][index]

                    # update the lower and upper bounds, and Jind
                    Jind += len(index)
                n2m_lower += self._batchsizeslist[j]

            return Jx, Jy

        # save the train jacobians per expert
        Jxtrain, Jytrain = jacobian_per_experts(True, Jind, numcounter,
                                                n2m_lower, nr_expert, self.bdata)
        self.Jsaveload.mode = "train_jacobians" + str(self.targetout)
        if self.savermode == 'zarr' or self.savermode == 'npy':
            self.Jsaveload.save_zarr(Jxtrain, None,
                                     Jytrain, None,
                                     nrexpert=nr_expert,
                                     is_verbose=False)
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

    def _jacobian_two_step(self, dataset):
        """ Jacobian computations for dataloader in a batch.

        Args:
            dataset (torch dataloader): pytorch dataloader (can be for both training and testing).
                                        this variable is passed to _jacobian_two_step.
        """
        # initial parameters
        self.max_batch_size = 5
        self._count, datacounter = 0, 0
        outputdim, adapt, do_repeat = True, True, True

        # jacobian saver loop
        while do_repeat:
            # set the batch size 
            loadingobject = torch.utils.data.DataLoader(MapDataset(dataset, DatasetMapper(self.detr_cfg, is_train=True)),
                                                        sampler=TrainingSampler(len(dataset)),
                                                        collate_fn=operator.itemgetter(0),
                                                        num_workers=0)
            dataloader = AspectRatioGroupedDataset(loadingobject, batch_size=int(self.max_batch_size))
            repeat = max(1, int(self.m / self.max_batch_size))

            # pushing the memory to the maximum
            try:
                logging.info("Updated batchsize: %s", str(self.max_batch_size))
                for batch_ndx, sample in enumerate(dataloader):
                    logging.info("Looping through: %s", str(batch_ndx))

                    # compute Jacobians
                    mq = DetectronQuantiles(model=self.model,
                                            data=sample,
                                            classout=self.classout,
                                            regout=self.regout,
                                            task=self.task,
                                            devices=self.args.device)
                    (Xhatsub, _, _, _, _) = mq.projection()

                    # skip the loop if Xhatsub is none
                    if Xhatsub is None:
                        continue

                    # size for collecting subset
                    xsize = int(Xhatsub.shape[1])
                    if not adapt:
                        datacounter += xsize

                    # converting to a float
                    if not self.is_float:
                        Xhatsub = Xhatsub.half()

                    # saving the number of parameters as a global variable
                    save_p = True
                    if save_p:
                        self.p = mq.p
                        save_p = False

                    # increasing the batch size
                    if adapt:
                        logging.info("Increasing batchsize by factor 4")
                        self.max_batch_size *= 4
                        if self.max_batch_size >= self.m:
                            self.max_batch_size /= 2
                            adapt = False
                        break

                    # save the last jacobians
                    if not adapt and int(datacounter) >= int(self.m):
                        logging.info("Saving the batch number: %s", str(batch_ndx))
                        if self._count > 0:
                            self.Jsaveload.mode = str(self.targetout) + "subset/"
                            cutind = int(xsize) - int(int(datacounter) - int(self.m))
                            logging.info("Saving the batch number after cutting down: %s", str(cutind))
                            self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0)[0:cutind],
                                                    None, None, None,
                                                    nrexpert=self._count)
                            self._count += 1
                        else:
                            self.Jsaveload.mode = str(self.targetout) + "subset/"
                            self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0),
                                                    None, None, None,
                                                    nrexpert=self._count)
                            self._count += 1

                        # stopping criteria for Jacobian compuations
                        do_repeat = False
                        logging.info("Stopping: %s bigger than %s",
                                     str(int((batch_ndx+1) * self.max_batch_size)),
                                     int(self.m))
                        break

                    # save all the jacobians
                    if not adapt and not int(datacounter) >= int(self.m):
                        logging.info("Saving the batch number: %s", str(batch_ndx))
                        self.Jsaveload.mode = str(self.targetout) + "subset/"
                        self.Jsaveload.save_ckp(Xhatsub.squeeze(dim=0),
                                                None, None, None,
                                                nrexpert=self._count)
                        self._count += 1

            except RuntimeError as run_error:
                # resize the batch size, turn off the adaptation and reassign for spoofing
                logging.info("Adaptation of the batch size %s", str(self.max_batch_size))
                self.max_batch_size /= 2.5 # 1.8 # TODO: determine this variable carefully.
                self.max_batch_size = int(self.max_batch_size)
                adapt = False
                repeat = max(1, int(self.m / self.max_batch_size))

                # Manual check if the RuntimeError was caused by the CUDA or something else
                logging.info(f"---\nRuntimeError: \n{run_error}\n---\n : adapting batch size")

    def _ntk_pca(self, dataset, is_train):
        """ Computes principle components of NTK
        Args:
            dataset (torch dataloader): pytorch dataloader (can be for both training and testing).
                                        this variable is passed to _jacobian_two_step.
            is_train (bool): if true, saves the jacobians on as train_jacobian, else test_jacobian.

        Returns:
            Xtransform (np.array): kernel PCA projected data.
        """
        # Initialization
        batchcounter, Xtransform = 0, list()
        if is_train:
            self._batchsizeslist =  list()

        # data loaders
        loadingobject = torch.utils.data.DataLoader(MapDataset(dataset, DatasetMapper(self.detr_cfg, is_train=True)),
                                                    sampler=InferenceSampler(len(dataset)), #shuffle=True,
                                                    collate_fn=operator.itemgetter(0),
                                                    num_workers=0)
        dataloader = AspectRatioGroupedDataset(loadingobject,
                                                batch_size=int(self.max_batch_size))

        for batch_ndx, sample in enumerate(dataloader):
            logging.info("Looping through: %s", str(batch_ndx))

            # Jacobian computations
            mq = DetectronQuantiles(model=self.model,
                                    data=sample,
                                    classout=self.classout,
                                    regout=self.regout,
                                    task=self.task,
                                    devices=self.args.device)
            (Xhat, yhat, _, _, _) = mq.projection()

            # skip the loop if Xhatsub is none
            if Xhat is not None:
                print("shape:", Xhat.shape)
                if is_train:
                    print("appending")
                    self._batchsizeslist.append(Xhat.shape[1])
                    print("_batchsizeslist", self._batchsizeslist)

                # set to half precision if possible
                if not self.is_float:
                    Xhat, yhat = Xhat.half(), yhat.half()

                # save the given variables
                self.Jsaveload.mode = "is_train" + str(is_train) + str(self.targetout)
                if self.is_zeroout:
                    Xhatpruend = Xhat.squeeze(dim=0)[:, self.pruneindex]
                    logging.info("Pruning results: from %s to %s", str(Xhat.shape), str(Xhatpruned.shape))
                    self.p_pruned = Xhatpruned.shape[1]
                    self.Jsaveload.save_numpy(Xhatpruned.cpu().detach().numpy(), None,
                                              yhat.squeeze(dim=0).cpu().detach().numpy(), None,
                                              nrexpert=batchcounter)
                else:
                    self.Jsaveload.save_numpy(Xhat.squeeze(dim=0).cpu().detach().numpy(), None,
                                              yhat.squeeze(dim=0).cpu().detach().numpy(), None,
                                              nrexpert=batchcounter)

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

                # update the counter
                batchcounter += 1

        # caching the batch
        if is_train:
            self._train_count = batchcounter
        else:
            self._test_count = batchcounter

        return np.concatenate(Xtransform)

    def _compute_theta_map(self, model):
        """Computes theta map - estimates of network weights with MAP principle.

        Returns:
            theta_map (torch.Tensor): the map estimates of parameters
        """
        theta_map = None
        for p in model.parameters():
            theta_map = torch.cat([p.flatten()])
        return theta_map 


class GaterDetectron(DivisionStepDetectron):
    """Single input, but multiple output gater function.

    Args:
        DivisionStepNeuralTangent ([type]): [description]
    """

    def __init__(self, traindata, testdata, model,
                 Jsaveload, args, detr_cfg, outdim=10,
                 saver=None):
        super().__init__(model=model, Jsaveload=Jsaveload, args=args, saver=saver,
                         classout=0, regout=0, task="regression", detr_cfg=detr_cfg, targetout=0)
        self.traindata = traindata
        self.testdata = testdata
        self.outdim = outdim

    def __call__(self):
        bdatalist = list()
        for targetout in range(self.outdim):
            # set the output dim number
            logging.info("Target output: %s", str(targetout))
            self.targetout = targetout
            if targetout==0:
                self.task="classification"
                self.classout=0
                self.regout=0
            elif targetout==5:
                self.task="classification"
                self.classout=1
                self.regout=0
            elif targetout>0 and targetout<5:
                self.task="regression"
                self.classout=0
                self.regout=targetout
            elif targetout>5 and targetout<10:
                self.task="regression"
                self.classout=1
                self.regout=targetout-6
            else:
                raise AttributeError

            # division step
            bdata = self.divisionstep(self.traindata, self.testdata)
            bdatalist.append(bdata)

            # use parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                logging.info("Saving Jacobians of experts")
                results = [executor.submit(self.expertsjacobiansaver, i) \
                        for i in range(self.n_clusters)]
                for f in concurrent.futures.as_completed(results):
                    print(f.result())
        return bdatalist

    def gating_function(self, test_jacobian, targetout):
        # targetout and setting up the variables
        self.targetout = targetout
        if targetout==0:
            self.task="classification"
            self.classout=0
            self.regout=0
            self._count=10
        elif targetout==5:
            self.task="classification"
            self.classout=1
            self.regout=0
            self._count=10
        elif targetout>0 and targetout<5:
            self.task="regression"
            self.classout=0
            self.regout=targetout
            self._count=10
        elif targetout>5 and targetout<10:
            self.task="regression"
            self.classout=1
            self.regout=targetout-6
            self._count=10
        else:
            raise AttributeError

        # load K-means and kPCA (TODO: later, load them once, and pass)
        filename = self.args.checkpoint_dir + '/kmeans' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        loaded_kmeans = pickle.load(open(filename, 'rb'))
        filename = self.args.checkpoint_dir + '/ntkpca' + str(self.n_clusters) + str(self.init_nr) \
            + str(self.targetout) + ".sav"
        loaded_kpca = pickle.load(open(filename, 'rb'))

        # compute the projected test_jacobians TODO: anyway to figure out a way to reduce this complexity.
        self.Jsaveload.mode = str(self.targetout) + "subset/"
        NTK = torch.cat([(1.0/self.delta) * \
            test_jacobian.squeeze(dim=0) @ self.Jsaveload.load_ckp(i)['Jtrain'].T \
                        for i in range(int(self._count))], 1).cpu().numpy()
        Xtransform = loaded_kpca.transform(NTK)

        # test labels via K-means
        testlabels = pairwise_distances_argmin(Xtransform, loaded_kmeans.cluster_centers_)

        # return the K-means results.
        return testlabels

