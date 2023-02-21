# py
from typing import Callable, List, Tuple, Optional
from math import pi

# nn & rl
from torch import Tensor, tensor, kron, matrix_exp, complex64, zeros, relu
from numpy import ndarray

# lib
from base.quantum import Operator
from base.action_space import RestrictedActionSpace
from base.env import Env


"""This file contains utility functions for the project."""


def create_env(reward_distribution: Optional[Tensor] = None) -> Env:
    """creates an Env with a RestrictedActionSpace and a J operator as described in Eisert et al."""
    num_players: int = 2
    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    mat: Tensor = matrix_exp(-1j * gamma * kron(D, D) / 2)
    J: Operator = Operator(mat=mat)
    env: Env = Env(num_players=num_players)
    env.action_space = RestrictedActionSpace()
    env.J = J
    if reward_distribution is not None:
        env.reward_distribution = reward_distribution
    return env


def calc_dist(nash_eq: Tensor, actions: Tensor, parametrization: Callable[[Tensor], Tensor]) -> Tensor:
    """calculates a distance between the specified nash equilibrium and the actions taken by players,
    given a parametrization which maps parameters to quantum operators."""
    dist: Tensor = tensor(0.)
    for i in range(0, nash_eq.shape[0]):
        nash_eq_op: Tensor = parametrization(nash_eq[i])
        params_op: Tensor = parametrization(actions[i])
        diff: Tensor = nash_eq_op - params_op
        dist += diff.abs().square().sum().sqrt()
    return dist


def compute_oriented_std(arr: Tensor, dim: int = 0) -> Tuple[ndarray, ndarray]:
    means: Tensor = arr.mean(dim)
    steps: int = arr.view(-1).shape[0] // arr.shape[dim]

    # allocate memory
    up_std: Tensor = zeros((steps,))
    down_std: Tensor = zeros((steps,))

    arr = arr.view(steps, -1)

    for idx, elem_tensor in enumerate(arr):
        ups: List[Tensor] = list()
        downs: List[Tensor] = list()
        for elem in elem_tensor:
            if elem >= means[idx]:
                ups.append(elem)
            else:
                downs.append(elem)
        up_std[idx] = tensor(ups).mean(0)
        down_std[idx] = tensor(downs).mean(0)

    return down_std.numpy(), up_std.numpy()


def compute_relu_std(arr: Tensor) -> Tuple[ndarray, ndarray]:
    upper_stand_dev_prep: Tensor = relu(arr - arr.mean(0)).mean(0)
    upper_stand_dev = list()
    for v in upper_stand_dev_prep:
        if v > 0.:
            upper_stand_dev.append(v)
    upper_stand_dev: ndarray = tensor(upper_stand_dev).numpy()
    lower_stand_dev_prep: Tensor = relu(arr.mean(0) - arr).mean(0)
    lower_stand_dev = list()
    for v in lower_stand_dev_prep:
        if v > 0.:
            lower_stand_dev.append(v)
    lower_stand_dev: ndarray = tensor(lower_stand_dev).numpy()
    stand_dev: Tuple[ndarray, ndarray] = (lower_stand_dev, upper_stand_dev)
    return stand_dev
