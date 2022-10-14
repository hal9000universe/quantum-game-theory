# py
from math import pi
from itertools import chain
from typing import Optional, Tuple

# nn & rl
from torch import tensor, Tensor, cat, kron, full, real, zeros, manual_seed
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch import float32, complex64
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Optimizer

# lib
from quantum import QuantumSystem, Ops, Operator, J


def rotation_operator(params: Tensor) -> Operator:
    """
    computes and returns a complex rotation matrix given by the angles in params.
    """
    theta, phi = params
    # calculate entries
    a: Tensor = exp(1j * phi) * cos(theta / 2)
    b: Tensor = sin(theta / 2)
    c: Tensor = -b
    d: Tensor = exp(-1j * phi) * cos(theta / 2)
    # construct the rows of the rotation matrix
    r1: Tensor = cat((a.view(1), b.view(1)))
    r2: Tensor = cat((c.view(1), d.view(1)))
    # build and return the rotation matrix
    rot: Tensor = cat((r1, r2)).view(2, 2)
    return Operator(mat=rot)


class Env:
    _state: Optional[QuantumSystem]
    _J: Operator
    _J_adj: Operator
    _rewards = Tensor

    def __init__(self):
        """
        initializes Env class:
         Env implements the quantum circuit described in
         Quantum games and quantum strategies
         by Eisert et al. 2020.
        """
        # state variables
        self._state = None
        # operators
        gamma: float = pi / 2
        D: Tensor = tensor([[0., 1.],
                            [-1., 0.]], dtype=complex64)
        mat = matrix_exp(-1j * gamma * kron(D, D) / 2)
        self._J = Operator(mat=mat)
        self._J_adj = Operator(mat=self._J.mat.adjoint())
        # reward distribution
        self._rewards = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

    @property
    def _ground_state(self) -> QuantumSystem:
        return QuantumSystem(num_qubits=2)

    def reset(self) -> Tensor:
        """prepares and returns the initial state according to Eisert et al. 2020."""
        self._state = self._J @ self._ground_state
        return self._state.state

    def step(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        calculates and returns the q-values for alice and bob given their respective actions.
        """
        # create rotation operators given by a1 and a2
        rot1, rot2 = rotation_operator(a1), rotation_operator(a2)
        # create operator
        op = rot1 + rot2
        # apply rotation operators to the quantum system
        self._state = op @ self._state
        # apply adjoint of J
        self._state = self._J_adj @ self._state
        # calculate expected reward respectively
        q1: Tensor = (self._rewards[:, 0] * self._state.probs).sum()
        q2: Tensor = (self._rewards[:, 1] * self._state.probs).sum()
        # return rewards
        return q1, q2


class ComplexNetwork(Module):
    _lin1: Linear
    _lin2: Linear
    _lin3: Linear
    _lin4: Linear
    _scaling: Tensor

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        """
        initializes ComplexNetwork class.
         ComplexNetwork implements a feed-forward neural network which is capable of handling 
         4-dimensional complex inputs. 
        """
        self._lin1 = Linear(4, 128, dtype=complex64)
        self._lin2 = Linear(128, 128, dtype=float32)
        self._lin3 = Linear(128, 128, dtype=float32)
        self._lin4 = Linear(128, 2, dtype=float32)
        self._scaling = tensor([pi, pi / 2])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        """
        Initializes weights using the kaiming-normal distribution and sets weights to zero.
        """
        if isinstance(m, Linear):
            manual_seed(2000)  # ensures reproducibility
            kaiming_normal_(m.weight)
            m.bias.data.fill_(0.)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = real(x)
        x = self._lin2(x)
        x = self._lin3(x)
        x = self._lin4(x)
        x = self._scaling * sigmoid(x)
        return x


def main():
    # initialize base classes
    env: Env = Env()
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()
    ac_optimizer: Optimizer = Adam(params=chain(alice.parameters(), bob.parameters()))

    # loop over episodes
    for step in range(1000):
        # get state from environment
        state = env.reset()

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)

        # compute q-values
        q_alice, q_bob = env.step(ac_alice, ac_bob)

        # define loss
        alice_loss = -q_alice
        bob_loss = -q_bob
        pg_loss = alice_loss + bob_loss

        # update agents
        ac_optimizer.zero_grad()
        pg_loss.backward()
        ac_optimizer.step()

        if step % 200 == 0:
            print(q_alice.item(), q_bob.item())
            print(ac_alice, ac_bob)
            print('---------')

    state = env.reset()
    ac_al = alice(state)
    ac_bo = bob(state)
    reward_a, reward_b = env.step(ac_al, ac_bo)
    print('actions after training: ')
    print('alice: {}'.format(ac_al))
    print('bob: {}'.format(ac_bo))
    print('reward after training: {} {}'.format(reward_a.item(), reward_b.item()))


if __name__ == '__main__':
    main()
