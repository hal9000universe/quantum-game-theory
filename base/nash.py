# py
from typing import List, Optional, Tuple
from copy import copy
from itertools import product

# nn & rl
from torch import Tensor, tensor

# lib
from base.env import Env


"""This file is about finding nash equilibria of quantum games by discretizing action spaces 
and looping over all possible action combinations."""


def is_nash(equilibrium: List, env: Env) -> bool:
    """The is_nash function checks if a set of strategies is the nash equilibrium of a quantum game.

    Keyword arguments
    equilibrium: a list of torch.Tensors specifying the set of strategies to be checked.
    env: Env specifying the quantum game, handling the game logic and computing the q-values."""
    # for all players i, for all actions s_i' (Q(*equilibrium) >= Q(s_1, ..., s_i', ..., s_n))
    for i, action in enumerate(equilibrium):
        # iterate of players
        eq = copy(equilibrium)
        max_q: Tensor = env.step(*eq)
        for try_action in env.action_space.iterator:
            # check if the i-th player could have done better by adjusting his strategy
            eq[i] = try_action
            q = env.step(*eq)
            if q[i] > max_q[i]:
                return False
    return True


def compute_nash_eq_b(env: Env) -> Optional[Tensor]:
    """The compute_nash_eq_b function tries to compute nash equilibria for quantum games."""
    action_space_grid: List[Tuple[Tensor, ...]] = []
    for action in env.action_space.iterator:
        action_space_grid.append(action)
    # create an iterable over all possible strategy combinations
    action_combinations = product(*[copy(action_space_grid) for _ in range(0, env.num_players)])
    for strats in action_combinations:
        strategies: List[Tuple[Tensor, ...]] = list(strats)
        if is_nash(strategies, env):
            return tensor(strategies)
