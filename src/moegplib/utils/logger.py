""" Creates a logging utilities such as check point saver and tensorboard summary writer.
"""
import torch
import os
import io
import numpy as np
import logging
import zarr

from shutil import copyfile
from numcodecs import Blosc
from torch.utils.tensorboard import SummaryWriter
from moegplib.utils.utils import read_pickle, write_pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class JacobiansSaveAndLoad(object):
    """Saves and loads Jacobians and related variables

    Args:
        object ([type]): [description]
    """
    def __init__(self, checkpth, expset, mode='default/', device='cuda', 
                 cname='lz4', clevel=1):
        """Initialize JacobiansSaveAndLoad

        Args:
            checkpth ([str]): [checkpoint directory]
            expset ([str]): [experiment number (given by clusters)]
            mode ([str]): [mode for saving jacobians (given by clusters)]
            device (str, optional): [device (cuda or cpu)]. Defaults to 'cuda'.
            cname (str, optional): A string naming one of the compression algorithms 
            available within blosc, e.g., ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’.
                BloscLZ - internal default.
                LZ4 - a compact, and fast.
                LZ4HC - better compression ratio but slower than LZ4.
                Snappy - a popular compressor.
                Zlib - classic but slower.
                Zstd - very balanced codec
                See https://catchchallenger.first-world.info/wiki/Quick_Benchmark:_Gzip_vs_Bzip2_vs_LZMA_vs_XZ_vs_LZ4_vs_LZO
            clevel (int, optional): An integer between 0 and 9 specifying the compression level. 
                Choose 1 (defulat) unless you lack disk storage
        """
        self.path = checkpth
        self.device = device
        self.expset = expset
        self.cname = cname
        self.clevel = clevel
        self.mode = mode
        self.n_path = self.path + '/' + self.expset + 'experts/'
        self._create_dirpth(self.path)
        self._create_dirpth(self.n_path)

    def _create_dirpth(self, dirpath, mkdirs=True):
        """Creates a directory if not existing

        We also catch filesystem error due to async operations.
        
        Args:
            dirpath ([str]): [a directory]
        """
        try:  
            if not os.path.isdir(dirpath):
                if mkdirs:
                    os.makedirs(dirpath)
                else:
                    os.mkdir(dirpath)
        except FileExistsError:
            print("Directory not created! Resuming...")
            
    def save_ckp(self, Jtrain, Jtest, yhat, Jbnd, nrexpert):
        """Saves the checkpoints.

        Args:
            Jtrain ([tensor]): [jacobians over the training points]
            Jtest ([tensor]): [jacobian over the test points]
            yhat ([tensor]): [transformed output for ntk gp]
            Jbnd ([tensor]): [jacobian over the boundary points]
            nrexpert ([int]): [number of expert]
        """        
        # save checkpoint data to the path given, checkpoint_path
        state = {
            'Jtrain': Jtrain,
            'Jtest': Jtest,
            'yhat': yhat,
            'Jbnd': Jbnd,
            'nrexpert': nrexpert
        }
        savepath = self.n_path + "/" + self.mode
        self._create_dirpth(savepath)
        torch.save(state, savepath + "/checkpoint" + str(state['nrexpert']) + "expert.pt")
        
    def load_ckp(self, nrexpert, targetout=None):
        """Load the check point given the expert number i

        Args:
            nrexpert ([int]): [the expert number i]

        Returns:
            [dictionary]: [a python dictionary state]
        """
        # only load boundary jacobian.
        if targetout is not None: 
            self.mode = "boundary_jacobian" + str(targetout)
            savepath = self.n_path + "/" + self.mode
            self._create_dirpth(savepath)
            return torch.load(savepath + "/checkpoint" + str(nrexpert) + "expert.pt")['Jbnd'].squeeze()
        else:
            savepath = self.n_path + "/" + self.mode
            self._create_dirpth(savepath)
            try:
                return torch.load(savepath + "/checkpoint" + str(nrexpert) + "expert.pt")
            except FileNotFoundError:
                return np.nan
    
    def save_zarr(self, Jtrain, Jtest, yhat, Jbnd,
                  nrexpert, is_verbose=False, mkdirs=True):
        """
        
        I will assume that these variables are given a numpy array.

        Args:
            Jtrain ([type]): [description]
            Jtest ([type]): [description]
            yhat ([type]): [description]
            Jbnd ([type]): [description]
            nrexpert ([type]): [description]
            is_verbose (bool, optional): [description]. Defaults to False.
        """
        # save path creation
        savepath = self.n_path + "/" + self.mode
        self._create_dirpth(savepath, mkdirs=mkdirs)
        filename = savepath + "/checkpoint"
        self._create_dirpth(filename, mkdirs=mkdirs)
        varname = str(nrexpert)
        
        # initialize the zarr structure and compressor
        store = zarr.DirectoryStore(filename + varname)
        compressor = Blosc(cname=self.cname, 
                           clevel=self.clevel,
                           shuffle=Blosc.BITSHUFFLE)
        group = zarr.hierarchy.group(store=store,
                                     overwrite=False,
                                     synchronizer=zarr.ThreadSynchronizer())
        
        # store the data
        if Jtrain is not None:
            zarrjtrain = group.zeros('Jtrain' + varname,
                                    shape=Jtrain.shape,
                                    dtype=Jtrain.dtype,
                                    compressor=compressor)
            zarrjtrain[...]=Jtrain[...]
            print(zarrjtrain.info) if is_verbose else None
            
        if Jtest is not None:
            zarrjtest = group.zeros('Jtest' + varname,
                                    shape=Jtest.shape,
                                    dtype=Jtest.dtype,
                                    compressor=compressor)
            zarrjtest[...]=Jtest[...]
            print(zarrjtest.info) if is_verbose else None
            
        if yhat is not None:
            zarryhat = group.zeros('yhat' + varname,
                                    shape=yhat.shape,
                                    dtype=yhat.dtype,
                                    compressor=compressor)
            zarryhat[...]=yhat[...]
            print(zarryhat.info) if is_verbose else None
            
        if Jbnd is not None:
            zarrjbnd = group.zeros('Jbnd' + varname,
                                    shape=Jbnd.shape,
                                    dtype=Jbnd.dtype,
                                    compressor=compressor)
            zarrjbnd[...]=Jbnd[...]
            print(zarrjbnd.info) if is_verbose else None
        
    def load_zarr(self, nrexpert, targetout, returnmode='all', 
                  is_tensor=True, use_cpu=True):
        """A loading function from zarr savers. 

        Args:
            nrexpert ([type]): [description]
            is_tensor (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description] TODO: load via cpu?
        """
        # save path creation
        if returnmode == 'Jtrain' or returnmode == 'yhat':
            self.mode = "train_jacobians" + str(targetout)
            
        elif returnmode == 'Jtest':
            self.mode = "test_jacobians" + str(targetout)
        elif returnmode == 'Jbnd':
            self.mode = "boundary_jacobian" + str(targetout)
        else:
            raise NotImplementedError
        savepath = self.n_path + "/" + self.mode
        self._create_dirpth(savepath)
        filename = savepath + "/checkpoint"
        varname = str(nrexpert)
        
        # load functions
        loader = zarr.convenience.open(filename + varname)
        # Jtrain
        if returnmode == 'Jtrain':
            try:
                Jtrain = loader['Jtrain' + varname]
            except KeyError:
                logging.info("Jtrain is None at expert %s", str(nrexpert))
                Jtrain = np.nan
                return Jtrain
            if is_tensor:
                if use_cpu:
                    return torch.from_numpy(Jtrain[:]).to('cpu')
                else:
                    return torch.from_numpy(Jtrain[:]).to(self.device)
            if not is_tensor:
                return Jtrain
        
        # Jtest
        if returnmode == 'Jtest':
            try:
                Jtest = loader['Jtrain' + varname] # FIXME: this is due to the distkae, hsould set to jtest
            except KeyError:
                logging.info("Jtest is None at expert %s", str(nrexpert))
                Jtest = np.nan
                return Jtest
            if is_tensor:
                if use_cpu:
                    return torch.from_numpy(Jtest[:]).to('cpu')
                else:
                    return torch.from_numpy(Jtest[:]).to(self.device)
            if not is_tensor:
                return Jtest
        
        # yhat
        if returnmode == 'yhat':   
            try:
                yhat = loader['yhat' + varname]
            except KeyError:
                logging.info("yhat is None at expert %s", str(nrexpert))
                yhat = np.nan
                return yhat
            if is_tensor:
                if use_cpu:
                    return torch.from_numpy(yhat[:]).to('cpu')
                else:
                    return torch.from_numpy(yhat[:]).to(self.device)
            if not is_tensor:
                return yhat
        
        # Jbnd
        if returnmode == 'Jbnd':
            try:
                Jbnd = loader['Jbnd' + varname]
            except KeyError:
                logging.info("Jbnd is None at expert %s", str(nrexpert))
                Jbnd = np.nan
                return Jbnd
            if is_tensor:
                if use_cpu:
                    return torch.from_numpy(Jbnd[:]).to('cpu')
                else:
                    return torch.from_numpy(Jbnd[:]).to(self.device)
            if not is_tensor:
                return Jbnd
        
        # load all as dictionaries
        if returnmode == 'all':
            # returning a tensor variable
            if is_tensor:
                return {'Jtrain': torch.from_numpy(Jtrain[:]).to(self.device), 
                        'Jtest': torch.from_numpy(Jtest[:]).to(self.device), 
                        'yhat': torch.from_numpy(yhat[:]).to(self.device), 
                        'Jbnd': torch.from_numpy(Jbnd[:]).to(self.device), 
                        'nrexpert': nrexpert}
                
            # returning zarr variables
            return {'Jtrain': Jtrain, 'Jtest': Jtest, 
                    'yhat': yhat, 'Jbnd': Jbnd,
                    'nrexpert': nrexpert}
    
    def save_numpy(self, Jtrain, Jtest, yhat, Jbnd,
                  nrexpert, is_verbose=False):
        """[summary]

        Args:
            Jtrain ([type]): [description]
            Jtest ([type]): [description]
            yhat ([type]): [description]
            Jbnd ([type]): [description]
            nrexpert ([type]): [description]
            is_verbose (bool, optional): [description]. Defaults to True.
        """
        # creating the path
        savepath = self.n_path + "/" + self.mode
        self._create_dirpth(savepath)
        
        # store the data
        if Jtrain is not None:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jtrain.npy"
            np.save(file, Jtrain)
            
        if Jtest is not None:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jtest.npy"
            np.save(file, Jtest)
            
        if yhat is not None:
            file = savepath + "/checkpoint" + str(nrexpert) + "yhat.npy"
            np.save(file, yhat)
            
        if Jbnd is not None:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jbnd.npy"
            np.save(file, Jbnd)
        
    def load_numpy(self, nrexpert, is_tensor=False):
        """[summary]

        Args:
            nrexpert ([type]): [description]
            is_tensor (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # save path init
        savepath = self.n_path + "/" + self.mode
        
        # load functions
        try:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jtrain.npy"
            Jtrain = np.load(file)
            if is_tensor:
                torch.from_numpy(Jtrain).to(self.device)
        except FileNotFoundError:
            Jtrain = None
        try:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jtest.npy"
            Jtest= np.load(file)
            if is_tensor:
                torch.from_numpy(Jtest).to(self.device)
        except FileNotFoundError:
            Jtest = None
        try:
            file = savepath + "/checkpoint" + str(nrexpert) + "yhat.npy"
            yhat = np.load(file)
            if is_tensor:
                torch.from_numpy(yhat).to(self.device)
        except FileNotFoundError:
            yhat = None
        try:
            file = savepath + "/checkpoint" + str(nrexpert) + "Jbnd.npy"
            Jbnd = np.load(file)
            if is_tensor:
                torch.from_numpy(Jbnd).to(self.device)
        except FileNotFoundError:
            Jbnd = None
            
        # returning zarr variables
        return {'Jtrain': Jtrain, 'Jtest': Jtest, 
                'yhat': yhat, 'Jbnd': Jbnd,
                'nrexpert': nrexpert}

    def save_tensor(self, tensor, tensorname):
        """[saves the tensor with the given name]

        Args:
            tensor ([tensor]): [tensor to be saved]
            tensorname ([str]): [name of the tensor to save]
        """
        torch.save(tensor, self.n_path + "/" + self.mode + "/checkpoint" + tensorname + ".pt")

    def load_tensor(self, tensorname):
        """[loads the tensor with the given name]

        Args:
            tensorname ([str]): [name of the tensor to load.]

        Returns:
            [tensor]: [loaded tensor from .pt file.]
        """
        return torch.load(self.n_path + "/" + self.mode + "/checkpoint" + tensorname + ".pt")

    def is_file(self, nrexpert, mode):
        """[checks if there is already a file]

        Args:
            nrexpert ([int]): [expert number i]

        Returns:
            [bool]: [True if existing, otherwise false.]
        """
        filename = self.n_path + "/" + mode + "/checkpoint" + str(nrexpert) + "expert.pt"
        return os.path.isfile(filename)


class SaveAndLoad(object):
    """ CheckLogger saves and loads the models checkpoints.
    We save (i) best state, (ii) latest state of network, 
    model and optimizer state dict, and additional info such as nr. epochs.
    """
    def __init__(self, checkpth, is_generic=False, device='cuda'):
        """[summary]

        Args:
            checkpth ([type]): [description]
            is_generic (bool, optional): [description]. Defaults to False.
            device (str, optional): [description]. Defaults to 'cuda'.
        """
        self.state = None
        self.path = checkpth
        self.is_generic = is_generic
        self.device = device
        
        # create directory if not existing
        if not os.path.isdir(checkpth):
            os.mkdir(checkpth)
    
    def _create_dirpth(self, dirpath):
        """Creates a directory if not existing

        We also catch filesystem error due to async operations.
        
        Args:
            dirpath ([str]): [a directory]
        """
        try:  
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
        except FileExistsError:
            print("Directory not created! Resuming...")
            
    def state_update(self, epoch, valid_loss, model, optimizer):
        """[summary]

        Args:
            epoch ([type]): [description]
            valid_loss ([type]): [description]
            model ([type]): [description]
            optimizer ([type]): [description]
        """
        self.state = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
            
    def save_ckp(self, is_best):
        """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model

        Args:
            is_best (bool): [description]
        """
        # chckpoitn path
        f_path = self.path + "/checkpoint.pt"

        # save checkpoint data to the path given, checkpoint_path
        torch.save(self.state, f_path)

        # if it is a best model, min validation loss
        if is_best:
            best_path = self.path + "/minval"

            # create directory for the best model if not existing
            if not os.path.isdir(best_path):
                os.mkdir(best_path)

            # copy that checkpoint file to best path given, best_model_path
            copyfile(f_path, best_path + "/minloss_checkpoint.pt")
            
    def load_ckp(self, is_best):
        """
        is_best: load the best model for inference
        model: model that we want to load checkpoint parameters into       
        optimizer: optimizer we defined in previous training

        Args:
            is_best (bool): [description]

        Returns:
            [type]: [description]
        """
        # set the file path
        if is_best:
            f_path = self.path + "/minval" + "/minloss_checkpoint.pt"
        f_path = self.path + "/checkpoint.pt"

        # load check point
        if self.is_generic:
            # Load ScriptModule from io.BytesIO object
            with open(f_path, 'rb') as f:
                buffer = io.BytesIO(f.read())

            # Load all tensors to the original device
            torch.jit.load(buffer)

            # Load all tensors onto CPU, using a device
            buffer.seek(0)
            torch.jit.load(buffer, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(f_path, map_location=lambda storage, loc: storage)
        return checkpoint
    
    def save_pickle(self, data, expset, expertnr):
        """[summary]

        Args:
            data ([type]): [description]
            expset ([type]): [description]
            expertnr ([type]): [description]
        """
        self._create_dirpth(self.path + expset)
        f_path = self.path + expset + "/pruneidx" + str(expertnr) + ".txt"
        write_pickle(f_path, data)
        
    def load_pickle(self, expset, expertnr):
        """[summary]

        Args:
            expset ([type]): [description]
            expertnr ([type]): [description]

        Returns:
            [type]: [description]
        """
        self._create_dirpth(self.path + expset)
        f_path = self.path + expset + "/pruneidx" + str(expertnr) + ".txt"
        return read_pickle(f_path)
    
    def save_gpytorch(self, gpmodel, expset, expertnr):
        # checkpoint path
        self._create_dirpth(self.path + expset)
        f_path = self.path + expset + "/gpcheckpoint" + str(expertnr) + ".pt"
        
        # save checkpoint data to the path given, checkpoint_path
        torch.save(gpmodel.state_dict(), f_path)
        
    def load_gpytorch(self, expset, expertnr):
        # checkpoint path
        self._create_dirpth(self.path + expset)
        f_path = self.path + expset + "/gpcheckpoint" + str(expertnr) + ".pt"
        
        # load the checkpoint data
        state_dict = torch.load(f_path)
        return state_dict
      
    def is_file(self, expset, expertnr):
        self._create_dirpth(self.path + expset)
        f_path = self.path + expset + "/gpcheckpoint" + str(expertnr) + ".pt"
        return os.path.isfile(f_path)


class BoardLogger(object):
    """ A basic wrapper class for customization of tensorboard.
    """
    def __init__(self, checkpth):
        self.path = checkpth
        self.writer = SummaryWriter(self.path + "/summary")
        
        # create directory if not existing
        if not os.path.isdir(checkpth):
            os.mkdir(checkpth)
            
    def lg_scalar(self, s_name, s_value, n_iter):
        self.writer.add_scalar(s_name, s_value, n_iter)
        
    def lg_figure(self, f_name, f_value):
        self.writer.add_figure(f_name, f_value)
        
    def lg_graph(self, model):
        self.writer.add_graph(model)
