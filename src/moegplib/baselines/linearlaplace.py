import os

import torch
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.special import logsumexp

from curvature.curvature.curvatures import KFAC, Diagonal, BlockDiagonal


def toy_data(visualize: bool = False):
    """Some toy data for testing similar to the 'in-between uncertainty' paper.

    Args:
        visualize: Whether to visualize the genrated data. Defaults to false.

    Returns:
        Torch tensors containing train and test set.
    """
    np.random.seed(45)
    frequency = -4
    phase = 0.4
    x = np.arange(-2, 2, 0.01)
    y = np.sin(frequency * x + phase)

    # First cluster
    x1 = np.random.uniform(-1.1, -0.8, 100)
    y1 = np.sin(frequency * x1 + phase) + np.random.normal(0, 0.1, 100)
    
    # Second cluster
    x2 = np.random.uniform(0.3, 0.9, 100)
    y2 = np.sin(frequency * x2 + phase) + np.random.normal(0, 0.1, 100)

    # Join clusters
    x_train = np.concatenate([x1, x2])
    y_train = np.concatenate([y1, y2])
    x_test = x
    y_test = y

    if visualize:
        plt.plot(x, y)
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.show()

    return (x_train, y_train), (x_test, y_test)


def sarcos(root: str):
    """The SARCOS inverse kinematics dataset (https://github.com/Kaixhin/SARCOS).

    Args:
        root: Path to directory containing "sarcos_inv.mat" and "sarcos_inf_test.mat".

    Returns:
        The SARCOS train and test set as Numpy arrays.
    """
    sarcos_inv = scipy.io.loadmat(os.path.join(root, "sarcos_inv.mat"))
    sarcos_inv_test = scipy.io.loadmat(os.path.join(root, "sarcos_inv_test.mat"))

    x_train = sarcos_inv["sarcos_inv"][:, :21]
    y_train = sarcos_inv["sarcos_inv"][:, 21:]
    x_test = sarcos_inv_test["sarcos_inv_test"][:, :21]
    y_test = sarcos_inv_test["sarcos_inv_test"][:, 21:]

    return (x_train, y_train), (x_test, y_test)


def kuka(root: str, part: int = 1):
    """The KUKA inverse kinematics dataset (https://github.com/fmeier/kuka-data)

    Args:
        root: Path to directory containing "kuka1_online.txt" and "kuka1_offline.txt".
              Same for part 2.
        part: KUKA consists of two parts, 1 and 2.

    Returns:
        The KUKA train and test set of the chosen dataset part as Numpy arrays.
    """
    train = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_online.txt"))
    test = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_offline.txt"))

    x_train = train[:, :21]
    y_train = train[:, 21:]
    x_test = test[:, :21]
    y_test = test[:, 21:]

    return (x_train, y_train), (x_test, y_test)


def get_model(weight_path: str = "",
              cuda: bool = False):
    """The model used for both SARCOS and KUKA experiments.

    Args:
        weight_path: Path to pre-trained weights. Returns untrained model if empty.
        cuda: Set to `True` if model was pre-trained on a GPU.

    Returns:
        PyTorch sequential model.
    """
    d_in = 21
    h = 200
    d_out = 7
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d_out))
    if weight_path:
        # models have a Dropout layer which is missing
        # in this model, so we need to renaim the last layer in
        # order to load his trained weights.
        try:
            model.load_state_dict(torch.load(weight_path,
                                             map_location=torch.device('cuda' if cuda else 'cpu')))
        except RuntimeError:
            state_dict = torch.load(weight_path,
                                    map_location=torch.device('cuda' if cuda else 'cpu'))["state_dict"]
            state_dict["8.weight"] = state_dict.pop("9.weight")
            state_dict["8.bias"] = state_dict.pop("9.bias")
            model.load_state_dict(state_dict)
    return model


def toy_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1))


def nll_from_samples(samples: np.ndarray,
                     targets: np.ndarray,
                     sigma: float = .5,
                     train_statistic: np.ndarray = None):
    """Computes the negative log-likelihood from samples obtained from the models
       output distribution (e.g. sampling laplace).

    Args:
        samples: Samples from the models output distribution, size TxNxD.
                 T: # of samples, N: # predicted data points, D: data dimensionality.
        targets: Targets for predictions, size NxD.
        sigma: Observation noise.
        train_statistic: Mean and std of training inputs and targets.

    Returns:
        The negative log-likelihood between the samples from the models output distribution
        and target distribution.
    """
    if train_statistic:
        x_train_mean, x_train_std, y_train_mean, y_train_std = train_statistic
        targets = targets * y_train_std + y_train_mean
        samples = samples * y_train_std + y_train_mean
    a = logsumexp(-.5 * sigma * (targets[np.newaxis, :] - samples) ** 2, axis=0)
    b = np.log(len(samples))
    c = .5 * np.log(2 * np.pi)
    d = .5 * np.log(sigma)
    return -np.mean(a - b - c + d, axis=0)


def nll_with_sampling(predictions: np.ndarray,
                      targets: np.ndarray,
                      variance: np.ndarray,
                      sigma: float = .5,
                      train_statistic: np.ndarray = None,
                      n_samples: int = 1000):
    """Computes the negative log-likelihood from deterministic model outputs (predictions)
       by sampling from a multivariate Gaussian with mean=predictions and variance computed
       beforehand by some method (e.g. linearization).

    Args:
        predictions: Determinstic model outputs, size NxD.
                     T: # of samples, N: # predicted data points.
        targets: Targets for predictions, size NxD.
        variance: Predictive variance for each data point, size Nx1.
        sigma: Observation noise.
        train_statistic: Mean and std of training inputs and targets.
        n_samples: # of samples to draw from the multivariate Gaussian.

    Returns:
        The negative log-likelihood between the samples drawn from the multivariate Gaussian
        and the target distribution.
    """
    samples = np.array([np.random.normal(loc=predictions, scale=np.sqrt(variance)) for _ in range(n_samples)])
    if train_statistic:
        x_train_mean, x_train_std, y_train_mean, y_train_std = train_statistic
        targets = targets * y_train_std + y_train_mean
        samples = samples * y_train_std + y_train_mean
    a = logsumexp(-.5 * sigma * (targets[np.newaxis, :] - samples) ** 2, axis=0)
    b = np.log(n_samples)
    c = .5 * np.log(2 * np.pi)
    d = .5 * np.log(sigma)
    return -np.mean(a - b - c + d, axis=0)


def get_fisher(model,
               x_train: np.ndarray,
               y_train: np.ndarray,
               criterion,
               estimator: str = "kfac",
               batch_size: int = 32):
    """Computes an approximation to the Fisher information matrix. Possible approximations are
       diagonal (diag), Kronecker factored (kfac) and block diagonal (full, fisher).

    Args:
        model: A pre-trained torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        y_train: Training targets, size NxC. C: # of output dimensions.
        criterion: A torch.nn loss.
        estimator: The Fisher approximation strategy.
        batch_size: Batch size for Fisher estimation.

    Returns:
        List of torch.Tensor of fisher approximations with one element per NN layer.
    """
    if estimator.lower() in ["fisher", "full", "block"]:
        fisher = BlockDiagonal(model)
    elif estimator.lower() in ["diag", "diagonal"]:
        fisher = Diagonal(model)
    elif estimator.lower() == "kfac":
        fisher = KFAC(model)
    else:
        raise ValueError
    dataset = torch.utils.data.dataset.TensorDataset(x_train, y_train)
    loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size)
    for x, y in tqdm(loader, postfix="Computing Fisher"):
        y_pred = model(x)
        dist = torch.distributions.Normal(y_pred, torch.tensor([.5]))
        for i in range(10):
            y_sample = dist.sample()
            y_sample.requires_grad = True
            loss = criterion(y_pred, y_sample)
            model.zero_grad()
            loss.backward(retain_graph=True)
            fisher.update(batch_size)
    return fisher


def get_empirical_fisher(model,
                         x_train: np.ndarray,
                         y_train: np.ndarray,
                         criterion,
                         estimator: str = "kfac",
                         batch_size: int = 32):
    """Computes an approximation to the empirical Fisher information matrix,
       where targets are not sampled from the models output distribution but taken
       directly from the data distribution. Possible approximations are
       diagonal (diag), Kronecker factored (kfac) and block diagonal (full, fisher).

    Args:
        model: A pre-trained torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        y_train: Training targets, size NxC. C: # of output dimensions.
        criterion: A torch.nn loss.
        estimator: The Fisher approximation strategy.
        batch_size: Batch size for Fisher estimation.

    Returns:
        List of torch.Tensor of fisher approximations with one element per NN layer.
    """
    if estimator.lower() in ["fisher", "full", "block"]:
        fisher = BlockDiagonal(model)
    elif estimator.lower() in ["diag", "diagonal"]:
        fisher = Diagonal(model)
    elif estimator.lower() == "kfac":
        fisher = KFAC(model)
    else:
        raise ValueError

    dataset = torch.utils.data.dataset.TensorDataset(x_train, y_train)
    loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size)
    for x, y in tqdm(loader, postfix="Computing emp. Fisher"):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        model.zero_grad()
        loss.backward()
        fisher.update(batch_size)
    return fisher


def laplace(model,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            criterion,
            add: float,
            multiply: float,
            estimator: str = "kfac",
            batch_size: int = 32):
    """Computes a Fisher information matrix approximation, inverts it and is used as
       covariance matrix in a multivariate Gaussian approximation to the models posterior,
       where the trained parameters are the Gaussians mean.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        y_train: Training targets, size NxC. C: # of output dimensions.
        x_test: Test inputs, size MxD.
        y_test: Test targets, size MxC.
        criterion: A torch.nn loss instance.
        add: Value added to the diagonal of the Fisher.
        multiply: The Fisher is multiplied by this value.
        estimator: The Fisher approximation strategy.
        batch_size: Batch size for Fisher estimation.

    Returns:
        Predictions on the test inputs, size TxMxC. T: # of posterior samples.
    """
    fisher = get_fisher(model, x_train, y_train, criterion, estimator, batch_size)
    fisher.invert(add, multiply)

    predictions = list()
    for _ in tqdm(range(1000), total=1000, postfix="Predicting"):
        fisher.sample_and_replace()
        predictions.append(model(x_test).detach().cpu().numpy().squeeze())
    return predictions


def get_grads(model):
    """Aggregates gradients of all linear layers in the model into a vector.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.

    Returns:
        The aggregated gradients.
    """
    weight_grads = list()
    bias_grads = list()
    for module in model.modules():
        if module.__class__.__name__ in ['Linear']:
            weight_grads.append(module.weight.grad.contiguous().view(-1))
            bias_grads.append(module.bias.grad)
    weight_grads.extend(bias_grads)
    return torch.cat(weight_grads)


def get_params(model):
    """Aggregates model parameters of all linear layers into a vector.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.

    Returns:
        The aggregated parameters.
    """
    weights = list()
    biases = list()
    for module in model.modules():
        if module.__class__.__name__ in ['Linear']:
            weights.append(module.weight.contiguous().view(-1))
            biases.append(module.bias)
    weights.extend(biases)
    return torch.cat(weights)


def get_gnn(model,
            x_train: torch.Tensor):
    """Computes the Generalized Gauss Newton matrix (GNN).

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.

    Returns:
        The GNN.
    """
    gnn = 0
    out = model(x_train)
    for o in tqdm(out, postfix="Computing GNN"):
        model.zero_grad()
        o.backward(retain_graph=True)
        grads = get_grads(model)
        gnn += torch.ger(grads, grads)
    return gnn


def make_p(model,
           omega: float):
    """Aggregates added regularization terms into a vector.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        omega: The regularization strength (prior variance).

    Returns:
        The regularization vector.
    """
    weight_p = list()
    bias_p = list()
    for module in model.modules():
        if module.__class__.__name__ in ['Linear']:
            w = module.weight
            b = module.bias
            wp = torch.ones(w.numel(), device=w.device) / (omega ** 2 / w.shape[0])
            bp = torch.ones(b.numel(), device=b.device)
            weight_p.append(wp)
            bias_p.append(bp)
    weight_p.extend(bias_p)
    return torch.cat(weight_p)


def get_cov(model,
            x_train: torch.Tensor,
            omega: float,
            sigma: float):
    """Computes the covarinace matrix as the inverse of the (regularized) GNN.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        omega: The additive regularization strength (prior variance).
        sigma: The multiplicative regularization strength (observation noise).

    Returns:
        The covarinace matrix.
    """
    gnn = get_gnn(model, x_train)
    p = make_p(model, omega)
    return torch.inverse(np.reciprocal(sigma ** 2) * gnn + torch.diag(p))


def linearized_laplace(model,
                       x_train: torch.Tensor,
                       x_test: torch.Tensor,
                       omega: float,
                       sigma: float):
    """Computes the predicition and predictive variance using the linearized inverse GNN.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        x_test: Test inputs, size MxD.
        omega: The additive regularization strength (prior variance).
        sigma: The multiplicative regularization strength (observation noise).

    Returns:
        The predictions and their predictive variance.
    """
    cov = get_cov(model, x_train, omega, sigma)

    model.eval()
    var_list = list()
    predictions = list()
    for x in tqdm(x_test, postfix="Predicting"):
        y = model(x)
        model.zero_grad()
        y.backward()
        grads = get_grads(model)
        var_list.append((sigma ** 2 + grads.t() @ cov @ grads).detach().cpu().numpy())
        predictions.append(y.detach().cpu().numpy().squeeze())
    return np.array(predictions), np.array(var_list)


def linear_laplace_diag(model,
                        x_train: torch.Tensor,
                        x_test: torch.Tensor,
                        omega: float,
                        sigma: float):
    """Computes the predicition and predictive variance using the diagonal linearized inverse GNN.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        x_test: Test inputs, size MxD.
        omega: The additive regularization strength (prior variance).
        sigma: The multiplicative regularization strength (observation noise).

    Returns:
        The predictions and their predictive variance.
    """
    # Compute GNN
    gnn = torch.zeros((7, len(get_params(model)))).to(model.device())
    out = model(x_train)
    for o in tqdm(out, disable=True):
        for i in range(7):
            model.zero_grad()
            o[i].backward(retain_graph=True)
            grads = get_grads(model)
            gnn[i] += grads ** 2
    cov = torch.reciprocal(sigma * gnn + omega)

    # Compute predictive variance
    variance = list()
    out = model(x_test)
    for o in tqdm(out, disable=True):
        tmp = list()
        for i, y in enumerate(o):
            model.zero_grad()
            y.backward(retain_graph=True)
            grads = get_grads(model)
            tmp.append((torch.sum(grads ** 2 * cov[i])).detach().cpu().numpy())
        variance.append(tmp)
    return out.detach().cpu().numpy(), np.array(variance)


def train(model,
          x_train: torch.Tensor,
          y_train: torch.Tensor,
          optimizer,
          criterion,
          epochs: int,
          shuffle: bool = True):
    """Trains a model on the training data using the specified optimizer and
       loss criterion.

    Args:
        model: A (pre-trained) torch.nn.Module or torch.nn.Sequential model.
        x_train: Training inputs, size NxD. N: # of samples, D: Input dimensions.
        y_train: Training targets, size NxC. C: # of output dimensions.
        optimizer: A torch.optim optimizer instance.
        criterion: A torch.nn loss instance.
        epochs: Number passes over the training data.
        shuffle: Whether to shuffle the data for each epoch.
    """
    model.train()

    for epoch in tqdm(range(epochs), total=epochs, postfix="Training"):
        if shuffle:
            idx = torch.randperm(len(x_train))
            x_train = x_train[idx]
            y_train = y_train[idx]

        y_pred = model(x_train)
        loss = criterion(y_train, y_pred)
        model.zero_grad()
        loss.backward()
        optimizer.step()


def run_toy(estimator: str = "ll",
            cuda: bool = False):
    """Runs the toy experiment on the toy dataset using the toy model.

    Args:
        estimator: The curvature estimation mode. Can be linear laplace (ll)
                   or one of `kfac`, `diag` or `fisher` for sampling laplace.
        cuda: Whether to compute on the GPU.
    """
    # Load model and data
    model = toy_model()
    (x_train, y_train), (x_test, y_test) = toy_data()

    # Cast Numpy arrays to PyTorch tensors
    x_train = torch.unsqueeze(torch.from_numpy(x_train).float(), dim=1)
    y_train = torch.unsqueeze(torch.from_numpy(y_train).float(), dim=1)
    x_test = torch.unsqueeze(torch.from_numpy(x_test).float(), dim=1)
    y_test = torch.unsqueeze(torch.from_numpy(y_test).float(), dim=1)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    if cuda:
        model.cuda()
        x_train.cuda()
        y_train.cuda()
        x_test.cuda()
        y_test.cuda()
        criterion.cuda()

    # Load or train model
    try:
        state_dict = torch.load(os.path.join(os.path.abspath(''), f"toy_model.pt"))
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        train(model, x_train, y_train, optimizer, criterion, epochs=20000)
        torch.save(model.state_dict(), os.path.join(os.path.abspath(''), f"toy_model.pt"))

    # Run linear or sampling laplace
    if estimator.lower() in ["ll", "linear", "linear laplace", "linear_laplace"]:
        predictions, variance = linearized_laplace(model, x_train, x_test, omega=4, sigma=.1)
        nll = nll_with_sampling(predictions, y_test.cpu().numpy().squeeze(), variance, sigma=.1)
    else:
        prediction_samples = laplace(model, x_train, y_train, x_test, y_test, criterion,
                                     add=100., multiply=10., estimator=estimator, batch_size=1)
        nll = nll_from_samples(prediction_samples, y_test.cpu().numpy().squeeze(), sigma=.1)
        predictions = np.mean(prediction_samples, axis=0)
        variance = np.var(prediction_samples, axis=0)

    # Visualization
    plt.plot(x_test.cpu().numpy().squeeze(), y_test.cpu().numpy().squeeze(), color="blue")
    plt.plot(x_test.cpu().numpy().squeeze(), predictions, color="red")
    plt.fill_between(x_test.cpu().numpy().squeeze(), predictions - np.sqrt(variance),
                     predictions + np.sqrt(variance), color="blue", alpha=0.3)
    plt.scatter(x_train.cpu().numpy().squeeze(), y_train.cpu().numpy().squeeze(), c='black', marker='x')
    plt.ylim(-2, 2)
    plt.show()


def run_sarcos_kuka(data_dir: str,
                    dataset: str,
                    weight_path: str,
                    add: float,
                    multiply: float,
                    estimator="kfac",
                    normalize=True,
                    cuda: bool = False):
    """Run experiment on SARCOS or KUKA dataset.

    Args:
        data_dir: Path to the dataset files. See dataset section.
        dataset: One of `sarcos`, `kuka1`, `kuka2`.
        weight_path: Path to the pre-trained weights.
        add: Value added to the diagonal of the Fisher.
        multiply: The Fisher is multiplied by this value.
        estimator: The Fisher approximation strategy.
        normalize: Whether to normalize the data.
        cuda: Whether to compute on the GPU.

    Returns:
        The average NLL.
    """
    # Load model and data
    model = get_model(weight_path)
    if dataset == "kuka1":
        (x_train, y_train), (x_test, y_test) = kuka(data_dir, part=1)
    elif dataset == "kuka2":
        (x_train, y_train), (x_test, y_test) = kuka(data_dir, part=2)
    elif dataset == "sarcos":
        (x_train, y_train), (x_test, y_test) = sarcos(data_dir)
    else:
        raise ValueError

    # Normalize data
    if normalize:
        x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
        y_train_mean, y_train_std = y_train.mean(axis=0), y_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        y_train = (y_train - y_train_mean) / y_train_std
        x_test = (x_test - x_train_mean) / x_train_std
        y_test = (y_test - y_train_mean) / y_train_std

    if normalize and "sarcos" in weight_path:
        train_statistics = (x_train_mean, x_train_std, y_train_mean, y_train_std)
    else:
        train_statistics = None

    # Cast Numpy arrays to PyTorch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    criterion = torch.nn.MSELoss(reduction="sum")

    if cuda:
        model.cuda()
        x_train.cuda()
        y_train.cuda()
        x_test.cuda()
        y_test.cuda()
        criterion.cuda()

    if estimator == "ll":
        predictions, variance = linear_laplace_diag(model, x_train, x_test, add, multiply)
        nll = nll_with_sampling(predictions, y_test, variance, .5, train_statistics)
    else:
        predictions = laplace(model, x_train, y_train, x_test, y_test, criterion, add, multiply, estimator)
        nll = nll_from_samples(predictions, y_test, .5, train_statistics)
        predictions = np.mean(predictions, axis=0)

    if normalize and "sarcos" in weight_path:
        predictions = predictions * y_train_std + y_train_mean
        y_test = y_test.cpu().numpy() * y_train_std + y_train_mean
    mse = np.mean((predictions - y_test) ** 2, axis=0)
