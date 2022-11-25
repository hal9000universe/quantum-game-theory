# py
from typing import Optional, List, Tuple
from math import sqrt, pi

# nn & rl
from torch import Tensor, tensor, kron, complex64, ones, linspace
from torch.distributions import Distribution, Uniform

# lib
from quantum import Ops, Operator, QuantumSystem


class ActionSpace:
    _ranges: List[Tuple[float, float]]
    _num_steps: int
    _num_params: int
    _ops: Ops

    def __init__(self, ranges: List[Tuple[float, float]]):
        self._ranges = ranges
        self._num_steps = 30
        self._num_params = len(ranges)
        self._ops = Ops()

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def iterator(self) -> List[Tensor]:
        itr: List[Tensor] = []
        for rng in self._ranges:
            space: Tensor = linspace(rng[0], rng[1], steps=self._num_steps)
            itr.append(space)
        return itr

    def operator(self, params: Tensor) -> Tensor:
        raise NotImplementedError


class GeneralActionSpace(ActionSpace):

    def __init__(self):
        # euler angle ranges: [-pi, pi], [0, pi], [-pi, pi]
        ranges: List[Tuple[float, float]] = [(-pi, pi), (0., pi), (-pi, pi)]
        super(GeneralActionSpace, self).__init__(ranges=ranges)

    def operator(self, params: Tensor) -> Tensor:
        """
        computes and returns a complex rotation matrix given by the Euler angles in params.
        """
        return self._ops.U.inj(*params).mat


class RestrictedActionSpace(ActionSpace):

    def __init__(self):
        # angle ranges given in Quantum Games and Quantum Strategies by Eisert et al.
        ranges: List[Tuple[float, float]] = [(0., pi), (0., pi / 2)]
        super(RestrictedActionSpace, self).__init__(ranges=ranges)

    def operator(self, params: Tensor) -> Tensor:
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
        return rot


class Env:
    _num_players: int
    _reward_distribution: Tensor
    _ops: Ops
    _J: Operator
    _adj_J: Operator
    _state: QuantumSystem
    _uniform: Uniform

    def __init__(self,
                 num_players: int,
                 reward_distribution: Tensor):
        self._num_players = num_players
        self._reward_distribution = reward_distribution
        self._ops = Ops()
        self._J = self._ops.J.inject(num_players=num_players)
        self._adj_J = Operator(self._J.mat.adjoint())
        self._state = QuantumSystem(num_qubits=num_players)
        self._uniform = Uniform(-0.25, 0.25)

    def reset(self, fix_inp: bool = False) -> Tensor:
        self._state = QuantumSystem(num_qubits=self._num_players)
        if fix_inp:
            fixed_inp: Tensor = 0.1 * ones((8,), dtype=complex64)
            return fixed_inp
        else:
            rand_inp = self._uniform.sample((8,)).type(complex64)
            return rand_inp

    def _operator(self, *params) -> Operator:
        """
        returns an operator specified by
        """
        raise NotImplementedError

    def _general_local_operator(self, theta1: Tensor, theta2: Tensor, theta3: Tensor) -> Operator:
        """
        returns a general local unitary operator specified by theta1, theta2 & theta3.
        """
        rot1 = self._ops.RotZ.inj(theta1).mat
        rot2 = self._ops.RotX.inject(theta2).mat
        rot3 = self._ops.RotZ.inj(theta3).mat
        op = rot3 @ rot2 @ rot1
        return Operator(mat=op)

    def step(self, *args) -> List[Tensor]:
        assert len(args) == self._num_players
        # prepare initial_state
        self._state = self._J @ self._state

        # create operator which applies the local unitary actions
        op: Operator = self._operator(*args[0])
        for i in range(1, len(args)):
            op = op + self._operator(*args[i])

        # apply operator
        self._state = op @ self._state
        # apply adjoint state preparation operator
        self._state = self._adj_J @ self._state
        # get probs
        probs = self._state.probs
        # get q-values
        qs: List = []
        for i in range(self._num_players):
            q_i = (self._reward_distribution[:, i] * probs).sum()
            qs.append(q_i)
        return qs
