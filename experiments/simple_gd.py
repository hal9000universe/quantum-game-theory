# py
from math import pi
from typing import Optional, Tuple, Callable, List

# nn & rl
from torch import tensor, Tensor, cat, kron, real, ones, complex64, float32, allclose
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam
from torch.distributions import Uniform, Distribution

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import QuantumSystem, Operator
from multi_env import MultiEnv


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


class PrisonersDilemmaEnv:
    _state: Optional[QuantumSystem]
    _J: Operator
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
        # reward distribution
        self._rewards = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

    @property
    def num_players(self) -> int:
        return 2

    @property
    def _ground_state(self) -> QuantumSystem:
        return QuantumSystem(num_qubits=2)

    def step(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        calculates and returns the q-values for alice and bob given their respective actions.
        """
        # prepare initial state
        self._state = self._J @ self._ground_state
        # create rotation operators given by a1 and a2
        rot1: Operator = rotation_operator(a1)
        rot2: Operator = rotation_operator(a2)
        # apply tensor product to create an operator which acts on the 2-qubit system
        op: Operator = rot1 + rot2
        # apply rotation operators to the quantum system
        self._state = op @ self._state
        # apply adjoint of J
        self._state = self._J.adjoint @ self._state
        # calculate q-values
        q1: Tensor = (self._rewards[:, 0] * self._state.probs).sum()
        q2: Tensor = (self._rewards[:, 1] * self._state.probs).sum()
        # return q-values
        return q1, q2


def generate_parameters() -> Tensor:
    uniform_theta: Distribution = Uniform(0., pi)
    uniform_phi: Distribution = Uniform(0., pi/2)
    return tensor([uniform_theta.sample((1,)), uniform_phi.sample((1,))], requires_grad=True)


def main(env: MultiEnv, episodes: int):
    # initialize params and optimizers
    players_params: List[Tensor] = [generate_parameters() for _ in range(0, env.num_players)]
    optimizers: List[Optimizer] = [Adam(params=[players_params[i]]) for i in range(0, env.num_players)]
    # loop over episodes
    for step in range(1, episodes):
        # general optimization
        for i in range(0, env.num_players):
            qs: List[Tensor] = env.step(*players_params)
            loss: Tensor = -qs[i]
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
    return players_params


def check(final_params: Tensor) -> bool:
    learned: bool = allclose(final_params, tensor([0., pi / 2]),
                             rtol=0.05,
                             atol=0.05)
    return learned


def sim_success_evaluation():
    nums: int = 0
    times: int = 100
    for time in range(times):
        final_params1, final_params2 = main(PrisonersDilemmaEnv(), 4000)
        print(f"final actions: {final_params1, final_params2}")
        if check(final_params1) and check(final_params2):
            print('training successful ...')
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


if __name__ == '__main__':
    sim_success_evaluation()
    # result: 0.7 success rate
