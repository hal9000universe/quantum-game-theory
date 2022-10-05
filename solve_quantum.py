from torch import tensor, Tensor, cat, exp, sin, cos, kron, complex64, matrix_exp, full, real, sigmoid, relu, zeros
from torch.nn import Module, Linear
from torch.optim import Adam
from itertools import chain
from math import pi
from random import uniform


def rotation_operator(a: Tensor) -> Tensor:
    theta, phi = a
    a = exp(1j * phi) * cos(theta / 2)
    b = sin(theta / 2)
    d = exp(-1j * phi) * cos(theta / 2)
    r1 = cat((a.view(1), b.view(1)))
    r2 = cat((-b.view(1), d.view(1)))
    rot = cat((r1, r2)).view(2, 2)
    return rot


class Env:

    def __init__(self):
        # state variables
        self._state = None
        self._ground_state = tensor([1., 0., 0., 0.], dtype=complex64)
        # construct gate to prepare initial state
        D = tensor([[0., 1.], [1., 0.]], dtype=complex64)
        gamma = pi / 2
        self._J = matrix_exp(-1j * gamma * kron(D, D) / 2)
        # rewards
        self._rewards = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

    def reset(self) -> Tensor:
        # prepare ground state
        self._state = self._J @ self._ground_state
        return self._state

    def step(self, a1: Tensor, a2: Tensor):
        rot1, rot2 = rotation_operator(a1), rotation_operator(a2)
        self._state = kron(rot2, rot1) @ self._state
        self._state = self._J.conj() @ self._state
        reward1 = (self._rewards[:, 0] * self._state.abs().square()).sum()
        reward2 = (self._rewards[:, 1] * self._state.abs().square()).sum()
        return reward1, reward2


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


class QNetwork(Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        self._linx1 = Linear(4, 100)
        self._linx1.weight.data = full((100, 4), 0.01, dtype=complex64)
        self._linx2 = Linear(100, 32)
        self._lina1 = Linear(4, 100)
        self._lina2 = Linear(100, 32)
        self._lin1 = Linear(32, 16)
        self._lin2 = Linear(16, 1)

    def __call__(self, x: Tensor, a: Tensor) -> Tensor:
        x = self._linx1(x)
        x = real(x)
        x = relu(x)
        x = self._linx2(x)
        a = self._lina1(a)
        a = relu(a)
        a = self._lina2(a)
        z = x + a
        z = self._lin1(z)
        z = relu(z)
        z = self._lin2(z)
        return z


def main():
    env = Env()
    alice = ComplexNetwork()
    q_alice = QNetwork()
    bob = ComplexNetwork()
    q_bob = QNetwork()

    q_optimizer = Adam(params=chain(q_alice.parameters(), q_bob.parameters()))
    ac_optimizer = Adam(params=chain(alice.parameters(), bob.parameters()))

    for step in range(100000):
        state = env.reset()

        ac_alice = alice(state)
        ac_bob = bob(state)

        rew_alice, rew_bob = env.step(ac_alice, ac_bob)

        # update q-networks
        ac = cat((ac_alice, ac_bob)).view(4)
        q_val_alice = q_alice(state, ac)
        q_val_bob = q_bob(state, ac)

        q_loss_alice = (q_val_alice - rew_alice).square()
        q_loss_bob = (q_val_bob - rew_bob).square()
        q_loss = q_loss_alice + q_loss_bob

        q_optimizer.zero_grad()
        q_loss.backward()  # alternatively retain_graph=True
        q_optimizer.step()

        # update agents
        ac_alice = alice(state)
        ac_bob = bob(state)

        ac = cat((ac_alice, ac_bob)).view(4)
        q_val_alice = q_alice(state, ac)
        q_val_bob = q_bob(state, ac)

        alice_loss = -q_val_alice
        bob_loss = -q_val_bob
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
