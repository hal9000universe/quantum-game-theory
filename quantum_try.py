# py
from math import pi
from random import uniform

# nn & rl
from torch import Tensor, tensor, kron, complex64, full, relu, dot
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
        self._reward_distribution = tensor([1., 3., 3., 6.])

    def reset(self) -> Tensor:
        self._system = TwoQubitSystem()
        self._system.prepare_state()
        return self._system.state

    def step(self, operator: Tensor) -> Tensor:
        matrix = kron(operator, operator)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        dif: Tensor = self._target - self._system.state
        reward = -self._scale_factor * dif.abs().square().sum()
        return reward

    def new_step(self, operator: Tensor) -> Tensor:
        matrix = kron(operator, operator)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        reward: Tensor = dot(self._reward_distribution, self._system.state.abs().square())
        return reward


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


if __name__ == '__main__':
    qnet = ComplexNetwork()
    env = Env()
    optimizer = Adam(qnet.parameters())
    for epi in range(1000000):
        inp = env.reset()
        params = qnet(inp)
        rot_mat = rotation_operator(*params)
        reward_signal = env.new_step(rot_mat)
        loss: Tensor = -reward_signal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epi % 10000 == 0:
            print(params)
            print(rot_mat)
            print(loss)
