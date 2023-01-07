# py
from typing import Callable
from math import pi

# nn & rl
from torch import Tensor, tensor, kron, matrix_exp, complex64

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
    env: Env = Env(num_players=num_players)
    env.action_space = RestrictedActionSpace()
    env.J = J
    return env


def calc_dist(nash_eq: Tensor, actions: Tensor, parametrization: Callable[[Tensor], Tensor]) -> Tensor:
    dist: Tensor = tensor(0.)
    for i in range(0, nash_eq.shape[0]):
        nash_eq_op: Tensor = parametrization(nash_eq[i])
        params_op: Tensor = parametrization(actions[i])
        diff: Tensor = nash_eq_op - params_op
        dist += diff.abs().square().sum().sqrt()
    return dist
