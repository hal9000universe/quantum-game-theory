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
from env import ActionSpace, RestrictedActionSpace, GeneralActionSpace
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


# TODO: translate checking is_nash to simplified algorithm
# TODO: speed up algorithm by combining coarse region search with exact nash-equilibrium search
