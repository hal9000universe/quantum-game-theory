# py
from typing import Optional, List, Tuple, Callable

# nn & rl
from torch import Tensor, tensor
from torch.distributions import Distribution, Uniform

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from env import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from quantum import QuantumSystem, Ops, Operator


def create_circuit(num_players: int) -> Callable:
    dev: Device = device('default.qubit', wires=num_players)
    all_wires: List[int] = [i for i in range(0, num_players)]
    J: Tensor = Ops().J.inject(num_players).mat

    @qnode(device=dev, interface='torch')
    def circuit(*operators: Tuple[Tensor, ...]) -> Tensor:
        QubitUnitary(J, wires=all_wires)
        for wire, (operator) in enumerate(operators):
            QubitUnitary(operator, wires=wire)
        QubitUnitary(J.adjoint(), wires=all_wires)
        return probs(all_wires)

    return circuit


class MultiEnv:
    _reward_distribution: Tensor
    _num_players: int
    _action_space: ActionSpace
    _ops: Ops
    _uniform: Uniform
    _circuit: Callable

    def __init__(self, num_players: int):
        self._num_players = num_players
        self._action_space = GeneralActionSpace()
        self._ops = Ops()
        self._uniform = Uniform(-1., 1.)
        self._circuit = create_circuit(num_players)
        self.generate_random()

    def generate_symmetric(self):
        pass

    def generate_random(self):
        epsilon: float = 1e-8
        shape: Tuple[int, int] = (pow(2, self._num_players), self._num_players)
        rew_dist: Tensor = self._uniform.sample(shape)
        self._reward_distribution = rew_dist / max(rew_dist.norm(), epsilon)

    def run(self, *args) -> List[Tensor]:
        operators: List[Tensor] = []
        for params in args:
            operators.append(self._action_space.operator(params))
        probabilities: Tensor = self._circuit(*operators)
        qs: List[Tensor] = []
        for i in range(0, self._num_players):
            q_i: Tensor = (self._reward_distribution[:, i] * probabilities).sum()
            qs.append(q_i)
        return qs


# TODO: understand league system of AlphaStar
