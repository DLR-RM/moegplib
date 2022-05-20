""" Applying compression results after pruning.

Modified scripts from https://github.com/microsoft/nni
"""

import logging
import torch

logger = logging.getLogger('torch apply compression')


def apply_compression_results(model, masks_file, map_location=None, is_verbose=False):
    """  Apply the masks from ```masks_file``` to the model
    Note: this API is for inference, because it simply multiplies weights with
    corresponding masks when this API is called.

    Args:
        model (torch.nn.Module): The model to be compressed
        masks_file (str): The path of the mask file
        map_location (str, optional): the device on which masks are placed, same to map_location in ```torch.load```.
            Defaults to None.
    """
    masks = torch.load(masks_file, map_location)
    for name, module in model.named_modules():
        if name in masks:
            # print("##anem##", name)
            try:
                module.weight.data = module.weight.data.mul_(masks[name]['weight'])
                if hasattr(module, 'bias') and module.bias is not None and 'bias' in masks[name]:
                    module.bias.data = module.bias.data.mul_(masks[name]['bias'])
            except AttributeError:
                if is_verbose:
                    logger.warning('Module.weight not existing. Trying module.module.weight.')
                module.module.weight.data = module.module.weight.data.mul_(masks[name]['weight'])
                if hasattr(module.module, 'bias') and module.module.bias is not None and 'bias' in masks[name]:
                    module.module.bias.data = module.module.bias.data.mul_(masks[name]['bias'])
                