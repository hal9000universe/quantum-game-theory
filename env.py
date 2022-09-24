# py
from typing import Callable, List, Tuple, Dict
from statistics import mean

# nn & rl
from torch import tensor, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal
from torch import no_grad

# lib
from structs import State, Operator
from replay_buffer import ReplayBuffer, sample_batch

# global
rewards: List[Tuple[float, float]] = [(1., 1.), (0., 5.), (5., 0.), (3., 3.)]


class Env:
    _state: State
    _op: Operator
    _replay_buffer: ReplayBuffer
    _rewards: List

    def __init__(self, replay_buffer: ReplayBuffer):
        self._state = State()
        self._op = Operator()
        self._replay_buffer = replay_buffer
        self._rewards = list()

    def reset(self) -> State:
        self._state = State()
        return self._state

    def step(self, theta_a: Tensor, phi_a: Tensor, theta_b: Tensor, phi_b: Tensor, state_representation: Tensor):
        # apply operator defined by theta_i, phi_i to the state of the system
        self._state._representation = self._op.rotate_qubits(theta_a, phi_a, theta_b, phi_b, self._state.representation)
        # measure the state
        measurement = self._state.measure()
        # assign a reward according to the rules of the game
        reward_alice, reward_bob = rewards[measurement]
        # add to replay buffer
        self._replay_buffer.add(
            state_representation,
            tensor([theta_a, phi_a]),
            tensor([theta_b, phi_b]),
            reward_alice,
            reward_bob
        )
        # monitoring
        self._rewards.append(reward_alice + reward_bob)
        if len(self._rewards) > 100:
            # remove old rewards
            self._rewards.pop(0)

    @property
    def average_reward(self) -> float:
        return mean(self._rewards)
