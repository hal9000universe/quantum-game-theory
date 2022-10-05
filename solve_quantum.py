# py
from math import pi
from itertools import chain
from random import uniform
from typing import Optional, Tuple

# nn & rl
from torch import tensor, Tensor, cat, kron, full, real, zeros
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch import float32, complex64
from torch.nn import Module, Linear
from torch.optim import Adam, Optimizer


def rotation_operator(a: Tensor) -> Tensor:
    # construct rotation matrix given by angles in a
    theta, phi = a
    a: Tensor = exp(1j * phi) * cos(theta / 2)
    b: Tensor = sin(theta / 2)
    d: Tensor = exp(-1j * phi) * cos(theta / 2)
    r1: Tensor = cat((a.view(1), b.view(1)))
    r2: Tensor = cat((-b.view(1), d.view(1)))
    rot: Tensor = cat((r1, r2)).view(2, 2)
    return rot


class Env:
    _state: Optional[Tensor]
    _ground_state: Tensor
    _J: Tensor
    _rewards = Tensor

    def __init__(self):
        # state variables
        self._state = None
        self._ground_state = tensor([1., 0., 0., 0.], dtype=complex64)
        # construct gate to prepare initial state according to Eisert et al. 2020
        D = tensor([[0., 1.], [1., 0.]], dtype=complex64)
        gamma = pi / 2
        self._J = matrix_exp(-1j * gamma * kron(D, D) / 2)
        # reward distribution
        self._rewards = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

    def reset(self) -> Tensor:
        # prepare initial state according to Eisert et al. 2020
        self._state = self._J @ self._ground_state
        return self._state

    def step(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        # create rotation operators given by a1 and a2
        rot1, rot2 = rotation_operator(a1), rotation_operator(a2)
        # apply rotation operators to the quantum system
        self._state = kron(rot1, rot2) @ self._state
        # apply adjoint of J
        self._state = self._J.conj().transpose(0, 1) @ self._state
        # calculate expected reward respectively
        reward1: Tensor = (self._rewards[:, 0] * self._state.abs().square()).sum()
        reward2: Tensor = (self._rewards[:, 1] * self._state.abs().square()).sum()
        # return rewards
        return reward1, reward2


class ComplexNetwork(Module):
    _lin1: Linear
    _lin2: Linear
    _lin3: Linear
    _lin4: Linear
    _scaling: Tensor

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        self._lin1 = Linear(4, 256, dtype=complex64)
        self._lin2 = Linear(256, 128)
        self._lin3 = Linear(128, 64, dtype=float32)
        self._lin4 = Linear(64, 2, dtype=float32)
        self._scaling = tensor([pi, pi / 2])

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = real(x)
        x = self._lin2(x)
        x = self._lin3(x)
        x = self._lin4(x)
        x = self._scaling * sigmoid(x)
        return x


def main():
    env: Env = Env()
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()

    ac_optimizer: Optimizer = Adam(params=chain(alice.parameters(), bob.parameters()))

    for step in range(100000):
        state = env.reset()

        ac_alice = alice(state)
        ac_bob = bob(state)

        rew_alice, rew_bob = env.step(ac_alice, ac_bob)

        # update agents
        alice_loss = -rew_alice
        bob_loss = -rew_bob
        pg_loss = alice_loss + bob_loss

        ac_optimizer.zero_grad()
        pg_loss.backward()
        ac_optimizer.step()

        if step % 1000 == 0:
            print(rew_alice.item(), rew_bob.item())
            print(ac_alice, ac_bob)
            print('---------')


if __name__ == '__main__':
    main()
