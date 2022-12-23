# py
from typing import Callable

# nn & rl
from torch import Tensor, load, zeros
from torch.nn import Module
from torch.utils.data import IterableDataset

# lib
from training.training import get_max_idx, get_model_path
from dataset.dataset import MicroQuantumTrainingDataset, get_mqt_path, compute_max_ds_idx
from base.utils import calc_dist
from base.action_space import RestrictedActionSpace


# accuracy: 0.72
def test_pretrained_model() -> float:
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator
    qt_ds: IterableDataset = load(get_mqt_path(compute_max_ds_idx(get_mqt_path) - 1))
    model: Module = load(get_model_path(get_max_idx()))
    num_successes: int = 0
    num_games: int = 0
    nash_eq: Tensor = zeros((2, 2))
    y: Tensor = zeros((2, 2))
    for i, (x, nash) in enumerate(qt_ds):
        nash_eq[i % 2] = nash
        y[i % 2] = model(*x)
        if i % 2 == 1:
            dist: Tensor = calc_dist(nash_eq, y, parametrization)
            print(f"{i}: {dist}")
            num_games += 1
            if dist < 0.2:
                num_successes += 1
    return num_successes / num_games


if __name__ == '__main__':
    accuracy: float = test_pretrained_model()
    print(f"Accuracy: {accuracy}")
