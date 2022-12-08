# py
from typing import List, Optional, Tuple
from copy import copy
from math import pi
from itertools import product

# nn & rl
from torch import Tensor, tensor, zeros
from torch import complex64, kron, matrix_exp

# lib
from quantum import Operator
from env import ActionSpace, RestrictedActionSpace
from multi_env import MultiEnv


def is_nash(equilibrium: List, env: MultiEnv) -> bool:
    # for all players i, for all actions s_i' (Q(*equilibrium) >= Q(s_1, ..., s_i', ..., s_n))
    for i, action in enumerate(equilibrium):
        eq = copy(equilibrium)
        max_q: Tensor = env.step(*eq)
        for try_action in env.action_space.iterator:
            eq[i] = try_action
            q = env.step(*eq)
            if q[i] > max_q[i]:
                return False
    return True


def compute_nash_eq_b(env: MultiEnv) -> List[Tuple[Tensor, ...]]:
    action_space_grid: List[Tuple[Tensor, ...]] = []
    for action in env.action_space.iterator:
        action_space_grid.append(action)
    action_combinations = product(*[copy(action_space_grid) for _ in range(0, env.num_players)])
    for strats in action_combinations:
        strategies: List[Tuple[Tensor, ...]] = list(strats)
        if is_nash(strategies, env):
            return strategies


if __name__ == '__main__':
    # define quantum game
    num_players: int = 2
    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    mat: Tensor = matrix_exp(-1j * gamma * kron(D, D) / 2)
    J: Operator = Operator(mat=mat)
    reward_distribution: Tensor = tensor([[6., 6.], [2., 8.], [8., 2.], [0., 0.]])
    action_space: ActionSpace = RestrictedActionSpace()

    # initialize env
    environment: MultiEnv = MultiEnv(num_players=num_players)
    environment.J = J
    environment.reward_distribution = reward_distribution
    environment.action_space = action_space

    nash_eq: List[Tensor] = [tensor([0., pi / 2]), tensor([0., pi / 2])]
    print(is_nash(equilibrium=nash_eq, env=environment))

    computed_nash_eq: List[Tuple[Tensor]] = compute_nash_eq_b(environment)
    print(computed_nash_eq)

    print(is_nash(computed_nash_eq, environment))
