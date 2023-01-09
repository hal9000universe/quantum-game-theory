# py
from typing import Callable, Optional, Tuple

# nn & rl
from torch import Tensor, load, zeros
from torch.nn import Module
from torch.utils.data import IterableDataset

# lib
from training.training import get_max_idx, get_model_path
from dataset.dataset import QuantumTrainingDataset, MicroQuantumTrainingDataset, get_mqt_path, compute_max_ds_idx
from base.utils import calc_dist
from base.action_space import RestrictedActionSpace

# plotting
from numpy import ndarray, linspace
from matplotlib.pyplot import plot, show, subplots, setp
from tikzplotlib import save


# accuracy: 0.52 - 0.72
def test_pretrained_model(model: Optional[Module] = None, monitoring: bool = False) -> float:
    if model is None:
        model = load(get_model_path(get_max_idx()))
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = load(get_mqt_path(124))
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
    qt_ds: IterableDataset = load(get_mqt_path(124))
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


def compute_pretrained_success_rate_data(monitoring: bool = False) -> Tuple[ndarray, ndarray]:
    max_idx: int = get_max_idx()
    rates: Tensor = zeros((max_idx + 1,))
    for idx in range(0, max_idx + 1):
        model = load(get_model_path(idx))
        # test model
        rates[idx]: float = test_pretrained_model(model)
        if monitoring:
            print(rates[idx])

    # return ndarrays
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = rates.numpy()
    return x, y


def compute_pretrained_performance_data(monitoring: bool = False) -> Tuple[ndarray, ndarray]:
    max_idx: int = get_max_idx()
    mean_distances: Tensor = zeros((max_idx + 1,))
    for idx in range(0, max_idx + 1):
        model = load(get_model_path(idx))
        # compute average distance to nash equilibria
        mean_distances[idx]: float = compute_performance(model).mean()
        if monitoring:
            print(mean_distances[idx].detach())

    # return ndarray
    x: ndarray = linspace(0, max_idx, max_idx + 1)
    y: ndarray = mean_distances.detach().numpy()
    return x, y


def make_plots():
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

    # show()

    save("experiments/plots/pretrained.tex")


if __name__ == '__main__':
    accuracy: float = test_pretrained_model()
    print(f"Accuracy: {accuracy}")
