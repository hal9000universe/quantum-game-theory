# py
from typing import List, Optional, Tuple
from copy import copy
from itertools import product

# nn & rl
from torch import Tensor, tensor

# lib
from base.env import Env


def is_nash(equilibrium: List, env: Env) -> bool:
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


def compute_nash_eq_b(env: Env) -> Optional[Tensor]:
    action_space_grid: List[Tuple[Tensor, ...]] = []
    for action in env.action_space.iterator:
        action_space_grid.append(action)
    action_combinations = product(*[copy(action_space_grid) for _ in range(0, env.num_players)])
    for strats in action_combinations:
        strategies: List[Tuple[Tensor, ...]] = list(strats)
        if is_nash(strategies, env):
            return tensor(strategies)
