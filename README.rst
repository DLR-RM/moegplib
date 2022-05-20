==========
MOEGPLIB: A Library for Mixtures of GP Experts.
==========

MOEGPLIB contains the code used for the CoRL submission:

**Trust your robots! Predictive uncertainty estimation of neural networks with scalable Gaussian Processes.**

The intended features are the followings.

1. Implementations of Mixtures of Gaussian Process Experts with Neural Tangent Kernel. 
2. Implementations of MC-dropout [1].
3. Implementations of approximate uncertainty propagation with MC-dropout [2].
4. Implementations of Laplace approximation [3-4].
5. Implementations of Linear uncertainty propagation with Laplace approximation [5, 12].
    
Some of these methods are used as a baseline, and others are the results of the project.

**Motivation:** Deep Neural Networks (DNNs) are wonderful tools for building predictors with large amounts of data, but notorious for delivering poor uncertainty estimates. On the other hand, Gaussian Processes (GPs) are theoretically elegant, and known to provide well-calibrated uncertainties. But, GPs does not scale well to big data-sets, and generally lacks predictive power as oppose to DNNs. To obtain a tool that has the best of both worlds, we started this project. As GP regression is analytically tractable, making its predictions just serious of matrix multiplications. We also note that, popular models Bayesian Neural Networks, and deep ensembles are hard to deploy them on a robot, as they require combining multiple predictions of DNNs.

**Important Note:** This repository is for reviewing only and a preliminary version, which we plan to open source. Illegal usages of the code, or its distribution can have legal consequences. After a thorough clean up, restructuring and testing, we will open-source the code, following the official procedures.

Installation Guide
===========

**Installations using conda environments**

To install the repository, navigate to the root directory and run:

.. code-block:: console

    $ conda env create -f environment.yml

To activate the newly created environment, run:

.. code-block:: console

    $ conda activate moegp

To check if the environment is setup correctly run:
 
.. code-block:: console

    $ conda env list

This list should comply with ``environment.yml`` file.

Due to hardware and driver incompatibility, it might be necessary to install GPyTorch and PyTorch separately. Make sure to install the correction version with a supported NVIDIA driver.

    - `Pytorch <https://pytorch.org/>`_
    - `Gpytorch <https://gpytorch.ai/>`_

**Manual installations**

An alternative is to install the required packages manually.

.. code-block:: console

    $ pip/conda install numpy scipy torchvision tqdm psutil scikit-learn gputil zarr pandas pyscaffold
    $ pip install torch/conda install pytorch
    $ pip install gpytorch
    $ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

To generate figures, install the following additional dependencies:

.. code-block:: console

    $ pip/conda install matplotlib seaborn statsmodels colorcet

Getting Started
===========

Setup
^^^^^^^^^^^^^^^^^^^^^

To install the code as a Python package go to the root directory and run:

.. code-block:: console

    $ python setup.py develop

We set up this project using PyScaffold.

Documentation
^^^^^^^^^^^^^^^^^^^^^

The `Sphinx <https://www.sphinx-doc.org/en/master/>`_ tool-chain is already set-up, for the code documentations.

To access the HTML documentation:

.. code-block:: console

    $ python setup.py docs
    $ cd docs/
    $ make html
    
Then, inside the ``build/html`` directory, there will be a file: ``index.html`` which can be opened with any browser.

To acces the PDF documentation:

.. code-block:: console

    $ cd docs/
    $ make latexpdf
    $ cd ../build/sphinx/latex/
    $ make
    
This should generate a PDF file called user_guide.pdf.

Minimalistic Guide to Developers
===========

**Overview of directory structure**

.. code-block:: console

    .
    +-- docs/
    +-- src/
        | +-- curvature/
        | +-- moegplib/
            | +-- baselines/
            | +-- clustering/
            | +-- datasets/
            | +-- lightnni/
            | +-- moegp/
            | +-- networks/
            | +-- utils/
    +-- tools/
        | +-- snelson/
        | +-- trainer/

**Overview of important directories**

    - ``src/moegplib/baselines``: Utility code for baselines
    - ``src/moegplib/datasets``: Data loader implementations
    - ``src/moegplib/networks``: Neural Network Models
    - ``src/moegplib/curvature/`` The code of the Laplace baselines submodule (only available after recursive pull)
    - ``tools/``: Tools to train and reproduce the results. There is one directory for each class of experiments.

On Reproducing the Results 
===========

**Snelson Experiments**

The command lines to reproduce the snelson experiments can be found below. To be executed at the ``root``.

.. code-block:: console

    $ python tools/patchgp.py --ckp_dir PATH_TO_NETWORK_CHECKPOINTS --data_dir PATH_TO_DATA
 
.. code-block:: console

    $ python tools/localgp.py --ckp_dir PATH_TO_NETWORK_CHECKPOINTS --data_dir PATH_TO_DATA
 
``patchgp.py`` runs the snelson experiment with patchwork prior, and ``localgp.py`` produces the results for a pure local GPs.
 

Further Readings
============

We recommend above literatures for further reading.

.. [1] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.
.. [2] Postels, Janis, et al. "Sampling-free epistemic uncertainty estimation using approximated variance propagation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
.. [3] MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. Neural computation, 4(3), 448-472.
.. [4] Ritter, H., Botev, A., & Barber, D. (2018, January). A scalable laplace approximation for neural networks. In 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings (Vol. 6). International Conference on Representation Learning.
.. [5] MacKay, David JC. "Information-based objective functions for active data selection." Neural computation 4.4 (1992): 590-604.
.. [6] Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.
.. [7] Microsft Research, "Neural Network Intelligence". 
.. [8] MacKay, David JC. "Information-based objective functions for active data selection." Neural computation 4.4 (1992): 590-604.
.. [9] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Summer school on machine learning. Springer, Berlin, Heidelberg, 2003.
.. [10] Khan, Mohammad Emtiyaz, et al. "Approximate inference turns deep networks into gaussian processes." Proceedings of Neural Information Processing Systems 32 (NeurIPS 2019)
.. [11] Jacot et al."Neural Tangent Kernel: Convergence and Generalization in Neural Networks." Proceedings of Neural Information Processing Systems 31 (NeurIPS 2018)
.. [12] Foong, Andrew YK, et al. "'In-Between'Uncertainty in Bayesian Neural Networks." arXiv preprint arXiv:1906.11537 (2019).

