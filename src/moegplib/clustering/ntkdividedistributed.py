""" This files contain division step for distriubted computing
"""
import torch
import scipy
import logging
import os
import operator
import concurrent.futures
import numpy as np

from torch.multiprocessing import Queue, Pool, Process, set_start_method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass
from moegplib.moegp.clustering.base import DivisionStepNeuralTangent, boundary_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class GaterDistributed(DivisionStepNeuralTangent):
    """Single input, but multiple output gater function.

    Args:
        DivisionStepNeuralTangent ([type]): [description]
    """
    
    def __init__(self, traindata, testdata, outdim, model,
                 Jsaveload, args, checkpoint, max_batch_size=500, saver=None):
        """[summary]

        Args:
            traindata ([type]): [description]
            testdata ([type]): [description]
            outdim ([type]): [description]
        """
        super().__init__(model, Jsaveload=Jsaveload, args=args, checkpoint=checkpoint, n_princomp=2,
                         init_nr=10, max_iter=100, tol=1e-3, random_state=None, max_batch_size=max_batch_size,
                         delta=1, targetout=0, init_method='k-means++', alpha=1,
                         saver=saver)
        self.traindata = traindata
        self.testdata = testdata
        self.outdim = outdim
        self.results_lst = Queue()

    def __call__(self, targetout, gpu_id, nr_worker_saveJacobian):
        print("\033[91m############### IN division step: output dim {} obtains GPU {}! ###############\033[0m".format(targetout, gpu_id))
        torch.cuda.set_device(gpu_id)
        # set the output dim number
        logging.info("Target output: %s", str(targetout))
        self.targetout = targetout
        
        # division step 
        bdata = self.divisionstep(self.traindata, self.testdata, dev=gpu_id)
        self.results_lst.put((targetout, bdata))
        
        # use parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=nr_worker_saveJacobian) as executor:
            logging.info("Saving Jacobians of experts")
            results = [executor.submit(self.expertsjacobiansaver, i) \
                    for i in range(self.n_clusters)]
            for f in concurrent.futures.as_completed(results):
                print(f.result())

        print("\033[91m############### IN division step: output dim {} FNISHED on GPU {}! ###############\033[0m".format(targetout, gpu_id))
        return (targetout, bdata) 
