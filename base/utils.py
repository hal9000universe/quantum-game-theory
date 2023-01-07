# py
from typing import Callable, Tuple
from math import pi

# nn & rl
from torch import Tensor, tensor, kron, matrix_exp, complex64, relu
from numpy import ndarray

# lib
from base.quantum import Operator
from base.action_space import RestrictedActionSpace
from base.env import Env


def create_env() -> Env:
    num_players: int = 2
    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    mat: Tensor = matrix_exp(-1j * gamma * kron(D, D) / 2)
    J: Operator = Operator(mat=mat)
    reward_distribution: Tensor = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])
    env: Env = Env(num_players=num_players)
    env.action_space = RestrictedActionSpace()
    env.J = J
    env.reward_distribution = reward_distribution
    return env


def calc_dist(nash_eq: Tensor, actions: Tensor, parametrization: Callable[[Tensor], Tensor]) -> Tensor:
    dist: Tensor = tensor(0.)
    for i in range(0, nash_eq.shape[0]):
        nash_eq_op: Tensor = parametrization(nash_eq[i])
        params_op: Tensor = parametrization(actions[i])
        diff: Tensor = nash_eq_op - params_op
        dist += diff.abs().square().sum().sqrt()
    return dist


def compute_oriented_std(arr: Tensor) -> Tuple[ndarray, ndarray]:
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
