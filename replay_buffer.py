# py
from typing import Tuple, List

# nn & rl
from torch import tensor, Tensor, zeros, randint


class ReplayBuffer:
    _buffer_size: int
    _states: Tensor
    _actions_alice: Tensor
    _actions_bob: Tensor
    _rewards_alice: Tensor
    _rewards_bob: Tensor
    _observations: Tensor
    _counter: int
    _num_samples: int

    def __init__(self,
                 buffer_size: int,
                 obs_shape: Tuple,
                 ac_shape: Tuple
                 ):
        self._buffer_size = buffer_size
        self._states = zeros(obs_shape)
        self._actions_alice = zeros(ac_shape)
        self._actions_bob = zeros(ac_shape)
        self._rewards_alice = zeros((buffer_size,))
        self._rewards_bob = zeros((buffer_size,))
        self._counter = 0
        self._num_samples = 0

    @property
    def states(self) -> Tensor:
        return self._states

    @property
    def actions_alice(self) -> Tensor:
        return self._actions_alice

    @property
    def actions_bob(self) -> Tensor:
        return self._actions_bob

    @property
    def rewards_alice(self) -> Tensor:
        return self._rewards_alice

    @property
    def rewards_bob(self) -> Tensor:
        return self._rewards_bob

    @property
    def size(self) -> int:
        return self._num_samples

    def add(self,
            state: Tensor,
            action_alice: Tensor,
            action_bob: Tensor,
            reward_alice: float,
            reward_bob: float,
            ):
        self._states[self._counter % self._buffer_size] = state
        self._actions_alice[self._counter % self._buffer_size] = action_alice
        self._actions_bob[self._counter % self._buffer_size] = action_bob
        self._rewards_alice[self._counter % self._buffer_size] = reward_alice
        self._rewards_bob[self._counter % self._buffer_size] = reward_bob
        self._counter += 1
        self._num_samples = min(self._counter, self._buffer_size)


def sample_batch(num_samples: int,
                 states: Tensor,
                 actions_alice: Tensor,
                 actions_bob: Tensor,
                 rewards_alice: Tensor,
                 rewards_bob: Tensor,
                 batch_size: int
                 ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # define indices in replay buffer which are going to be sampled
    random_indices: Tensor = randint(high=num_samples, size=(batch_size,))
    # create batch containing states, the actions taken by alice and bob, as well as the subsequently received rewards
    batch = (
        states[random_indices],
        actions_alice[random_indices],
        actions_bob[random_indices],
        rewards_alice[random_indices],
        rewards_bob[random_indices],
    )
    return batch
