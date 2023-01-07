# py
from typing import Callable, Optional

# nn & rl
from torch import Tensor, load, zeros, relu
from torch.nn import Module
from torch.utils.data import IterableDataset

# lib
from training.training import get_max_idx, get_model_path
from dataset.dataset import MicroQuantumTrainingDataset, get_mqt_path, compute_max_ds_idx
from base.utils import calc_dist
from base.action_space import RestrictedActionSpace

# plotting
from numpy import ndarray, linspace
from matplotlib.pyplot import plot, show, errorbar
from tikzplotlib import save


# accuracy: 0.52 - 0.72
def test_pretrained_model(model: Optional[Module] = None, monitoring: bool = False) -> float:
    if model is None:
        model = load(get_model_path(get_max_idx()))
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = load(get_mqt_path(compute_max_ds_idx(get_mqt_path) - 1))
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


def compute_performance(model: Optional[Module] = None, monitoring: bool = False) -> Tensor:
    if model is None:
        model = load(get_model_path(get_max_idx()))
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = load(get_mqt_path(compute_max_ds_idx(get_mqt_path) - 1))
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


def plot_pretrained_success_rate():
    max_idx: int = get_max_idx()
    rates: Tensor = zeros((max_idx + 1,))
    for idx in range(0, max_idx + 1):
        model = load(get_model_path(idx))
        # test model
        rates[idx]: float = test_pretrained_model(model)
        print(rates[idx])

    # plot data
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = rates.numpy()
    plot(x, y, c="b")
    show()

    # save plot
    save("experiments/plots/pretrained-success-rate.tex")


def plot_pretrained_performance():
    max_idx: int = get_max_idx()
    mean_distances: Tensor = zeros((max_idx + 1, 200))
    for idx in range(0, max_idx + 1):
        model = load(get_model_path(idx))
        # compute average distance to nash equilibria
        mean_distances[idx]: float = compute_performance(model)
        print(mean_distances[idx].mean().detach())

    # plot data
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = mean_distances.mean(1).detach().numpy()
    plot(x, y, c="b")
    show()

    # save plot
    save("experiments/plots/pretrained-performance.tex")


if __name__ == '__main__':
    plot_pretrained_performance()
    plot_pretrained_success_rate()
