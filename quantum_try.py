# py
from math import pi
from random import uniform
from typing import Tuple

# nn & rl
from torch import Tensor, tensor, kron, complex64, full, relu, dot, tensordot
from torch import real, exp, sin, cos, view_as_real, sigmoid, cat, stack
from torch.nn import Module, Linear
from torch.optim import Adam
from scipy.stats import unitary_group

# lib
from quantum import TwoQubitSystem, Ops


class Env:

    def __init__(self):
        self._system = TwoQubitSystem()
        assert self._system.check_conditions()
        self._target = tensor([0., 0., 0., 1j], dtype=complex64)
        self._scale_factor = tensor(100., dtype=complex64)
        self._reward_distribution = tensor([6., 3., 3., 1.])
        self._smart_reward_distribution = tensor([3., 0., 5., 1.])
        self._multi_reward_distribution = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]], requires_grad=True).transpose(0, 1)

    def reset(self) -> Tensor:
        self._system = TwoQubitSystem()
        self._system.prepare_state()
        return self._system.state

    def step(self, operator: Tensor) -> Tensor:
        matrix = kron(operator, operator)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        self._system.prepare_state()
        dif: Tensor = self._target - self._system.state
        reward = -self._scale_factor * dif.abs().square().sum()
        return reward

    def rl_step(self, operator: Tensor) -> Tensor:
        matrix = kron(operator, operator)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        self._system.prepare_state()
        reward: Tensor = dot(self._reward_distribution, self._system.state.abs().square())
        return reward

    def smart_step(self, operator: Tensor) -> Tensor:
        smart_matrix = tensor([[1j, 0.], [0., -1j]], dtype=complex64)
        matrix = kron(operator, smart_matrix)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        self._system.prepare_state()
        reward: Tensor = dot(self._smart_reward_distribution, self._system.state.abs().square())
        return reward

    def multi_step(self, alice_operator: Tensor, bob_operator: Tensor) -> Tuple[Tensor, Tensor]:
        matrix = kron(alice_operator, bob_operator)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        self._system.prepare_state()
        probs: Tensor = self._system.state.abs().square()
        rewards: Tensor = probs * self._multi_reward_distribution
        reward_alice, reward_bob = rewards.sum(dim=1)
        return reward_alice, reward_bob


class ComplexNetwork(Module):

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        self._lin1 = Linear(4, 64)
        self._lin1.weight.data = full((64, 4), 0.01, dtype=complex64)
        self._lin2 = Linear(64, 32)
        self._lin2.weight.data = full((32, 64), 0.1, dtype=complex64)
        self._lin3 = Linear(32, 2)
        self._lin3.weight.data = full((2, 32), 1., dtype=complex64)
        self._scaling = tensor([pi, pi / 2])

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = self._lin2(x)
        x = self._lin3(x)
        x = real(x)
        x = self._scaling * sigmoid(x)
        return x


def rotation_operator(theta: Tensor, phi: Tensor) -> Tensor:
    m11: Tensor = (exp(1j * phi) * cos(theta / 2)).unsqueeze(0)
    m12: Tensor = sin(theta / 2).unsqueeze(0)
    m21: Tensor = -m12
    m22: Tensor = (exp(-1j * phi) * cos(theta / 2)).unsqueeze(0)
    rotation_matrix = stack((cat((m11, m12)), cat((m21, m22))), dim=0)
    return rotation_matrix


def base():
    qnet = ComplexNetwork()
    env = Env()
    optim = Adam(qnet.parameters())
    for epi in range(100000):
        inp = env.reset()
        params = qnet(inp)
        rot_mat = rotation_operator(*params)
        reward = env.step(rot_mat)
        loss: Tensor = -reward
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epi % 1000 == 0:
            print(loss.item())
            print(params)


def reinforce():
    qnet = ComplexNetwork()
    env = Env()
    optim = Adam(qnet.parameters())
    for epi in range(100000):
        inp = env.reset()
        params = qnet(inp)
        rot_mat = rotation_operator(*params)
        reward = env.rl_step(rot_mat)
        loss: Tensor = -reward
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epi % 1000 == 0:
            print(params)


def smart_opponent():
    qnet = ComplexNetwork()
    env = Env()
    optim = Adam(qnet.parameters())
    for epi in range(100000):
        inp = env.reset()
        params = qnet(inp)
        rot_mat = rotation_operator(*params)
        reward = env.smart_step(rot_mat)
        loss: Tensor = -reward
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epi % 1000 == 0:
            print(params)


def multi():
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()
    env = Env()
    optim = Adam([{'params': alice.parameters()}, {'params': bob.parameters()}])
    for epi in range(1000000):
        inp = env.reset()
        alice_params = alice(inp)
        bob_params = bob(inp)
        alice_rot_mat = rotation_operator(*alice_params)
        bob_rot_mat = rotation_operator(*bob_params)
        rew_alice, rew_bob = env.multi_step(alice_rot_mat, bob_rot_mat)
        loss = - (rew_alice + rew_bob)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epi % 1000 == 0:
            print(alice_params, bob_params)


if __name__ == '__main__':
    smart_opponent()
