# py
from typing import List, Tuple, Callable, Optional

# nn & rl
from torch import Tensor, complex64, kron, tensor
from torch.distributions import Distribution, Uniform

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from base.action_space import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from base.quantum import QuantumSystem, Ops, Operator


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
    _uniform: Distribution
    _reward_sampler: Distribution
    _circuit: Callable
    _state: QuantumSystem
    _J: Operator
    _nash_eq: Optional[Tensor]

    def __init__(self, num_players: int):
        self._num_players = num_players
        self._action_space = GeneralActionSpace()
        self._ops = Ops()
        self._uniform = Uniform(-0.25, 0.25)
        self._circuit = create_circuit(num_players)
        self.generate_random()
        self._state = QuantumSystem(self._num_players)
        self._J = self._ops.J.inject(self._num_players)
        self._nash_eq = None

    @property
    def nash_eq(self) -> Optional[Tensor]:
        return self._nash_eq

    @nash_eq.setter
    def nash_eq(self, nash_eq: Optional[Tensor]):
        self._nash_eq = nash_eq

    @property
    def num_players(self) -> int:
        return self._num_players

    @num_players.setter
    def num_players(self, num_players: int):
        self._num_players = num_players

    @property
    def reward_distribution(self) -> Tensor:
        return self._reward_distribution

    @reward_distribution.setter
    def reward_distribution(self, reward_distribution: Tensor):
        # ensure correct shapes
        assert len(reward_distribution.shape) == 2
        assert pow(reward_distribution.shape[1], 2) == reward_distribution.shape[0]
        # normalize
        epsilon: float = 1e-8
        self._reward_distribution = reward_distribution / max(reward_distribution.norm(), epsilon)
        self._num_players = self._reward_distribution.shape[1]

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, operator: Operator):
        self._J = operator

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: ActionSpace):
        self._action_space = action_space

    def generate_random(self):
        shape: Tuple[int, int] = (pow(2, self._num_players), self._num_players)
        rew_dist: Tensor = self._uniform.sample(shape)
        self.reward_distribution = rew_dist

    def generate_random_symmetric(self):
        rewards: Tensor = self._uniform.sample((4,))
        rew_dist: Tensor = tensor([[rewards[0], rewards[0]], [rewards[1], rewards[2]], [rewards[2], rewards[1]],
                                   [rewards[3], rewards[3]]])
        self.reward_distribution = rew_dist

    def _observation(self) -> Tensor:
        return (self._J @ QuantumSystem(num_qubits=self._num_players)).state

    def reset(self, fix_inp: bool) -> Tensor:
        if fix_inp:
            return self._observation()
        else:
            rand_inp = self._uniform.sample((4,)).type(complex64)
            return self._observation() + rand_inp

    def _create_operator(self, args: Tuple[Tensor]) -> Operator:
        op: Tensor = self._action_space.operator(args[0])
        for i in range(1, len(args)):
            op = kron(op, self._action_space.operator(args[i]))
        return Operator(mat=op)

    def step(self, *args) -> List[Tensor]:
        # prepare initial_state
        self._state = self._J @ QuantumSystem(num_qubits=self._num_players)
        # create operator which applies the local unitary actions
        op = self._create_operator(args)
        # apply operator
        self._state = op @ self._state
        # apply adjoint state preparation operator
        self._state = self._J.adjoint @ self._state
        # get q-values
        qs: List = []
        for i in range(self._num_players):
            q_i = (self._reward_distribution[:, i] * self._state.probs).sum()
            qs.append(q_i)
        return qs

    def q_step(self, *args) -> List[Tensor]:
        ops: List[Tensor] = []
        for params in args:
            op: Tensor = self._action_space.operator(params)
            ops.append(op)
        probabilities: Tensor = self._circuit(*ops)
        # get q-values
        qs: List = []
        for i in range(self._num_players):
            q_i = (self._reward_distribution[:, i] * probabilities).sum()
            qs.append(q_i)
        return qs
