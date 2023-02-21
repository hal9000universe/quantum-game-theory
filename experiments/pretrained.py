# py
from typing import Callable, Optional, Tuple

# nn & rl
from torch import Tensor, load, zeros
from torch.nn import Module
from torch.utils.data import IterableDataset

# lib
from training.training import get_max_idx, path_fun_map
from dataset.dataset import QuantumTrainingDataset, MicroQuantumTrainingDataset, get_mixed_mqt_path, compute_max_ds_idx
from base.utils import calc_dist
from base.action_space import RestrictedActionSpace

# plotting
from numpy import ndarray, linspace
from matplotlib.pyplot import plot, show, subplots, setp
from tikzplotlib import save


"""This file is about testing a pre-trained transformer on quantum games."""


# accuracy: 0.52 - 0.72
def test_pretrained_model(model: Optional[Module] = None, monitoring: bool = False,
                          path_fun: Callable[[int], str] = get_mixed_mqt_path) -> float:
    """The test_pretrained_model function evaluates how well a transformer generalized from
    the provided quantum game data by testing on an independent quantum game data batch
    whether the transformer chooses actions close to the nash equilibrium"""
    if model is None:
        model_path_fun: Callable[[int], str] = path_fun_map(get_mixed_mqt_path)
        model: Module = load(model_path_fun(get_max_idx(model_path_fun)))
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = QuantumTrainingDataset(start=0.9, end=1., path_fun=path_fun)
    num_successes: int = 0
    num_games: int = 0
    nash_eq: Tensor = zeros((2, 2))
    y: Tensor = zeros((2, 2))
    for i, (x, nash) in enumerate(qt_ds):
        nash_eq[i % 2] = nash
        y[i % 2] = model(*x)
        if i % 2 == 1:
            dist: Tensor = calc_dist(nash_eq, y, parametrization)
            num_games += 1
            if dist < 0.2:
                num_successes += 1
            if monitoring:
                print(f"{i}: {dist}")
    return num_successes / num_games


def compute_performance(model: Optional[Module] = None, monitoring: bool = False,
                        path_fun: Callable[[int], str] = get_mixed_mqt_path) -> Tensor:
    """The compute_performance function computes the distances of the actions the pre-trained model
    chooses and the nash-equilibria."""
    if model is None:
        model_path_fun: Callable[[int], str] = path_fun_map(path_fun)
        model = load(model_path_fun(get_max_idx(model_path_fun)))
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = QuantumTrainingDataset(start=0.9, end=1., path_fun=path_fun)
    num_games: int = len(qt_ds)
    distances: Tensor = zeros((num_games,))
    nash_eq: Tensor = zeros((2, 2))
    y: Tensor = zeros((2, 2))
    for i, (x, nash) in enumerate(qt_ds):
        nash_eq[i % 2] = nash
        y[i % 2] = model(*x)
        if i % 2 == 1:
            distances[i // 2]: Tensor = calc_dist(nash_eq, y, parametrization)
            if monitoring:
                print(f"{i}: {dist}")
    return distances


def compute_pretrained_success_rate_data(monitoring: bool = False,
                                         path_fun: Callable[[int], str] = get_mixed_mqt_path
                                         ) -> Tuple[ndarray, ndarray]:
    """The compute_pretrained_success_rate_data function returns an x,y pair of coordinates
    for a matplotlib.pyplot graph. The data represents the success rate of the agent depending
    on the number of training epochs."""
    model_path_fun: Callable[[int], str] = path_fun_map(path_fun)
    max_idx: int = get_max_idx(model_path_fun)
    rates: Tensor = zeros((max_idx + 1,))
    for idx in range(0, max_idx + 1):
        model = load(model_path_fun(idx))
        # test model
        rates[idx]: float = test_pretrained_model(model, path_fun=path_fun)
        if monitoring:
            print(rates[idx])

    # return ndarrays
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = rates.numpy()
    return x, y


def compute_pretrained_performance_data(monitoring: bool = False,
                                        path_fun: Callable[[int], str] = get_mixed_mqt_path
                                        ) -> Tuple[ndarray, ndarray]:
    """The compute_pretrained_performance_data function returns an x,y pair of coordinates
    for a matplotlib.pyplot graph. The data represents the distances of the actions the agents choose
    to the nash-equilibria depending on the number of training epochs."""
    model_path_fun: Callable[[int], str] = path_fun_map(path_fun)
    max_idx: int = get_max_idx(model_path_fun)
    mean_distances: Tensor = zeros((max_idx + 1,))
    for idx in range(0, max_idx + 1):
        model = load(model_path_fun(idx))
        # compute average distance to nash equilibria
        mean_distances[idx]: float = compute_performance(model, path_fun=path_fun).mean()
        if monitoring:
            print(mean_distances[idx].detach())

    # return ndarray
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = mean_distances.detach().numpy()
    return x, y


def make_plots():
    """The make_plots function generates a tikz-file showing how the success rate of the transformer rises
    during the pre-training process."""
    # set up plot
    nrows: int = 1
    ncols: int = 2
    fig, axs = subplots(nrows, ncols)
    fig.set_figheight(5.)
    fig.set_figwidth(12.)

    # generate data
    x_performance, y_performance = compute_pretrained_performance_data(True)
    print("Performance Evaluation Done.")
    x_success_rate, y_success_rate = compute_pretrained_success_rate_data(True)
    print("Success Rate Evaluation Done.")

    # plot success rate data
    axs[0].plot(x_success_rate, y_success_rate)
    axs[0].set_title("Performanz")

    # plot performance data
    axs[1].plot(x_performance, y_performance)
    axs[1].set_title("Distanz zum Nash-Gleichgewicht")

    # label axes
    setp(axs, xlabel="Epochen")
    setp(axs[0], ylabel="Erfolgsrate")
    setp(axs[1], ylabel="d(s, s')")

    save("experiments/plots/pretrained.tex")


if __name__ == '__main__':
    accuracy: float = test_pretrained_model(path_fun=get_mixed_mqt_path)
    print(f"Accuracy: {accuracy}")
