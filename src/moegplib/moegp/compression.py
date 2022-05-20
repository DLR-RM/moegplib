"""A file that contains pruner, and other compressors.
"""

import torch
import logging
import copy

from moegplib.lightnni.algorithms import LevelPruner, SlimPruner, FPGMPruner, L1FilterPruner, \
    L2FilterPruner, AGPPruner, ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner, \
    TaylorFOWeightFilterPruner
from moegplib.lightnni.pruning import apply_compression_results
from moegplib.lightnni.speedup.compressor import ModelSpeedup
from moegplib.lightnni.utils.counter import count_flops_params


class JacobianPruner:
    """[summary]
    """
    def __init__(self, sparsity, masks_file=None, mode='zeroesout'):
        """[summary]
        Args:
            sparsity ([type]): [description]
            masks_file ([type], optional): [description]. Defaults to None.
            mode (str, optional): [description]. Defaults to 'zeroesout'.
        """
        self.sparsity = sparsity
        self.mode = mode
        self.masks_file = masks_file
        
    def neuralprune(self, Jacobian):
        """[summary]
        Args:
            Jacobian ([type]): [description]
        Raises:
            AttributeError: [description]
        """
        # check if masks_file is not None
        if self.masks_file is None:
            raise AttributeError
        
        # loading the masks file
        masks = torch.load(self.masks_file)
        
        # vectorizing the masks file            
        binary_masks = list()
  
        for i in masks:
            binary_masks.append(masks[i]['weight'].flatten())
            binary_masks.append(masks[i]['bias'].flatten())
        binary_masks = torch.cat(binary_masks)
        
        # calculating the indices
        pruneindx = torch.nonzero(binary_masks).squeeze()
        
        return copy.deepcopy(Jacobian)[:, pruneindx]
    
    def expertsprune(self, Jacobian, do_pruneindx=False):
        """[summary]
        Args:
            Jacobian ([type]): it has to be a torch tensor, N x p 
            with N number of data points, and p the number of parameters. 
        Returns:
            [type]: [description]
        """
        # absolute value and sum
        Jabs = torch.abs(Jacobian)
        Jsum = torch.sum(Jabs, dim=0)

        # ranking them and save the indices
        if self.mode == 'zeroesout': 
            pruneindx = torch.nonzero(Jsum)
        elif self.mode == 'sparse':
            try:
                ranking = torch.argsort(Jsum)
                pruneindx = ranking[int(self.sparsity*len(ranking)):]
            except ValueError:
                if do_pruneindx:
                    return Jacobian[:, torch.nonzero(Jsum)], torch.nonzero(Jsum)
                return Jacobian
        else:
            print("not implemented error")
            exit(0)
        
        # cut them.
        if do_pruneindx:
            return Jacobian[:, pruneindx], pruneindx
        return Jacobian[:, pruneindx]


class NeuralPruner:
    """Neural Pruner implementation
    """
    def __init__(self, args, sparsity, pruner='level', initial_sparsity=0.,
                 final_sparsity=0.8, start_epoch=0, end_epoch=10, frequency=1):
        """Initialization
        
        Args:
            model (torch.nn.Module): Pytorch model class
            args (parse): argument file. Need args.checkpoint_dir defined.
            sparsity (float): the sparsity ratio. 1.0 means prune all, 0.0 means prune nothing.
            pruner (str, optional): pruning method. 'taylor' and 'level supported. Defaults to 'level'.
        """
        self.args = args
        self.pruner = pruner
        if self.pruner == 'level' or self.pruner == 'taylor':
            self.is_oneshot = True
            self.is_iterative = False
        else:
            self.is_oneshot = False
            self.is_iterative = True
        self.prune_config = {
            'level': {
                'pruner_class': LevelPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Linear'],
                }]
            },
            'taylor': {
                'pruner_class': TaylorFOWeightFilterPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d'],
                    'op_names': ["conv1","conv2","conv3","conv4","conv5","conv6"]
                }]
            },
            'taylor': {
                'pruner_class': TaylorFOWeightFilterPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d'],
                }]
            },
            'agp': {
                'pruner_class': AGPPruner,
                'config_list': [{
                    'initial_sparsity': initial_sparsity,
                    'final_sparsity': final_sparsity,
                    'start_epoch': start_epoch,
                    'end_epoch': end_epoch,
                    'frequency': frequency,
                    'op_types': ['Conv2d']
                }]
            },
            'slim': {
                'pruner_class': SlimPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['BatchNorm2d']
                }]
            },
            'fpgm': {
                'pruner_class': FPGMPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d']
                }]
            },
            'l1filter': {
                'pruner_class': L1FilterPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d'],
                    'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
                }]
            },
            'mean_activation': {
                'pruner_class': ActivationMeanRankFilterPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d'],
                    'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
                }]
            },
            'apoz': {
                'pruner_class': ActivationAPoZRankFilterPruner,
                'config_list': [{
                    'sparsity': sparsity,
                    'op_types': ['Conv2d'],
                    'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
                }]
            }
        }
        
    def oneshot(self, model, inputsize, dummy_data=None, dependency_aware=False,
                applyresults=True, speedup=True, saver=None):
        """ Implementation of one-shot pruner.
        
        Args:
            model (Torch.module.nn): pytorch model class.
            inputsize (tuple): a python tuple describing the input size
                              (either use inputsize or give dummy_data).
            dummy_data (Torch.tensor, optional): an example input data. Defaults to None.
            dependency_aware (bool, optional): set dependency_aware mode to True.
                                               This should be set to True for filter based pruning methods (e.g. taylor)
                                               Defaults to False
            applyresults (bool, optional): Apply the masking to the network. Defaults to True.
            speedup (bool, optional): Apply the speed up using torch.jit. Defaults to True.
            saver (str, optional): Name to save the masks. Defaults to None.
            
        Returns:
            model_clone (Torch.module.nn): pytorch model class, pruned.
        """
        # introspection        
        if self.is_oneshot == False:
            print("This is not a one shot pruner. Exiting..")
            exit(0)
        
        # return without pruning if sparsity is 0%
        if self.prune_config[self.pruner]['config_list'][0]['sparsity'] == 0.0:
            return model
        
        # model clone
        model_clone = copy.deepcopy(model)
        
        # model compression
        if dependency_aware:
            if dummy_data is None:
                dummy_data = torch.ones(inputsize).to(self.args.device)
            pruner = self._create_pruner(model_clone, self.pruner, 
                                         dependency_aware=dependency_aware,
                                         dummy_input=dummy_data,
                                         statistics_batch_num=8) # FIXME: number to 8
        else:
            pruner = self._create_pruner(model_clone, self.pruner, dependency_aware=dependency_aware)
        model_clone = pruner.compress()
        
        # saving the checkpoints
        if saver is not None:
            model_path =  self.args.checkpoint_dir + '/' + saver + 'pruned_dnn.pth'
            mask_path =  self.args.checkpoint_dir + '/' + saver + 'mask_pruned_dnn.pth'
            pruner.export_model(model_path=model_path, 
                                mask_path=mask_path)
        else:
            model_path = self.args.checkpoint_dir + 'pruned_dnn.pth'
            mask_path = self.args.checkpoint_dir + 'mask_pruned_dnn.pth'
            pruner.export_model(model_path=model_path, 
                                mask_path=mask_path)
        
        # applying compression results
        if applyresults:
            apply_compression_results(model_clone, mask_path)
        
        # apply speed up
        if speedup:
            if dummy_data is not None:
                m_speedup = ModelSpeedup(model_clone, dummy_data, mask_path)
                m_speedup.speedup_model()
            else:
                dummy_data = torch.ones(inputsize).to(self.args.device)
                m_speedup = ModelSpeedup(model_clone, dummy_data, mask_path)
                m_speedup.speedup_model()
        
        return model_clone
        
    def iterative(self):
        raise NotImplementedError
    
    def _create_pruner(self, model, pruner_name, optimizer=None,
                       dependency_aware=False, dummy_input=None,
                       statistics_batch_num=None):
        """Creates the pruner.
        
        Args:
            model (Torch.module.nn): pytorch model class.
            pruner_name (str): name of the pruner.
            optimizer (Torch.optimizer, optional): Only need for iterative pruning. Defaults to None.
            dependency_aware (bool, optional): Set the dependency aware mode. Defaults to False.
            dummy_input (torch.tensor, optional): an example of input data. Defaults to None.
        """
        pruner_class = self.prune_config[pruner_name]['pruner_class']
        config_list = self.prune_config[pruner_name]['config_list']
        kw_args = {}
        if dependency_aware:
            # note that, not all pruners support the dependency_aware mode
            kw_args['dependency_aware'] = True
            kw_args['dummy_input'] = dummy_input
            kw_args['statistics_batch_num'] = statistics_batch_num # FIXME: only for taylor
        pruner = pruner_class(model, config_list, optimizer, **kw_args)
        return pruner