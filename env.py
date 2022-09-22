# py
from typing import Callable, List, Tuple, Dict

# nn & rl
from torch import tensor, Tensor, normal
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal
from torch import no_grad

# lib
from structs import State, Operator
from replay_buffer import ReplayBuffer, sample_batch

# global
rewards: List[Tuple[float, float]] = [(1., 1.), (0., 5.), (5., 0.), (3., 3.)]


def sample(params: Tensor) -> Tensor:
    mean: Tensor = params[0:2]
    std: Tensor = params[2:4]
    action: Tensor = normal(mean, std)
    return action


class Env:
    _state: State
    _op: Operator
    _replay_buffer: ReplayBuffer

    def __init__(self, replay_buffer: ReplayBuffer):
        self._state = State()
        self._op = Operator()
        self._replay_buffer = replay_buffer

    def episode(self, alice: Module, bob: Module):
        state = self._state.long()  # save state.long() before operator is applied
        theta_a, phi_a = sample(alice(self._state.long()))
        theta_b, phi_b = sample(bob(self._state.long()))
        self._state._representation = self._op.act(theta_a, phi_a, theta_b, phi_b, self._state.representation)
        measurement = self._state.measure()
        reward_alice, reward_bob = rewards[measurement]
        self._replay_buffer.add(
            state,
            tensor([theta_a, phi_a]),
            tensor([theta_b, phi_b]),
            reward_alice,
            reward_bob
        )
