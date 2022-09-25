# py
# from math import exp
from random import uniform

# nn & rl
from torch import Tensor, tensor, kron, complex64, full, relu, real, exp, sin, cos, view_as_real
from torch.nn import Module, Linear
from torch.optim import Adam
from scipy.stats import unitary_group

# lib
from quantum import TwoQubitSystem, Ops


class Env:

    def __init__(self):
        self._system = TwoQubitSystem()
        assert self._system.check_conditions()
        self._target = tensor([0., 0., 0., 1.], dtype=complex64)
        self._scale_factor = tensor(100., dtype=complex64)
        self._max_reward = 100.

    def reset(self) -> Tensor:
        self._system = TwoQubitSystem()
        self._system.prepare_state()
        return self._system.state

    def step(self, operator: Tensor) -> Tensor:
        matrix = kron(operator, operator).type(complex64)
        self._system.state = self._system.ops.general.inject(matrix, self._system.state)
        dif = (self._scale_factor * (self._target - self._system.state)).abs().square().sum()
        reward = tensor(self._max_reward, dtype=complex64, requires_grad=True) - dif
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

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = self._lin2(x)
        x = self._lin3(x)
        return x


def apply_params(theta: Tensor, phi: Tensor, state: Tensor):
    theta_half: Tensor = theta / 2
    a: Tensor = exp(2j * phi) * cos(theta_half).square()
    b: Tensor = exp(1j * phi) * cos(theta_half) * sin(theta_half)
    c: Tensor = cos(theta_half).square()
    d: Tensor = sin(theta_half).square()
    e: Tensor = exp(-2j * phi) * cos(theta_half).square()
    out1 = a * state[0] + b * state[1] + b * state[2] + d * state[3]
    out2 = -b * state[0] + c * state[1] + (-d) * state[2] + b * state[3]
    out3 = -b * state[0] + (-d) * state[1] + c * state[2] + b * state[3]
    out4 = d * state[0] + b * state[1] + (-b) * state[2] + e * state[3]
    return out1, out2, out3, out4


if __name__ == '__main__':
    qnet = ComplexNetwork()
    inp = tensor([0., 0., 0., -1.], dtype=complex64)
    target = tensor([0., 0., 0., 1.], dtype=complex64)
    optimizer = Adam(qnet.parameters())
    i_unit: Tensor = tensor(1j, dtype=complex64)
    for epi in range(1000000):
        params = qnet(inp)
        vec = apply_params(*params, inp)
        l = (target[0] - vec[0]).abs().square()
        l += (target[1] - vec[1]).abs().square()
        l += (target[2] - vec[2]).abs().square()
        l += (target[3] - vec[3]).abs().square()
        optimizer.zero_grad()
        l.backward()
        for param in qnet.parameters():
            print(param.shape)
            print(param.grad)
            exit()
        if epi % 1000 == 0:
            print(l.item())
