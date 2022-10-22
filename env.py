# py
from typing import Optional, List
from math import sqrt

# nn & rl
from torch import Tensor, tensor, kron, complex64, ones
from torch.distributions import Distribution, Uniform

# lib
from quantum import Ops, Operator, QuantumSystem


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


class QuantumSuperiorityGame(Env):

    def __init__(self):
        num_players: int = 3
        reward_distribution: Tensor = tensor([[0., 0., 0.],  # 1
                                              [-9, -9., 1.],  # 4
                                              [-9., 1., -9.],  # 3
                                              [1., 9., 9.],  # 5
                                              [1., -9., -9.],  # 2
                                              [9., 1., 9.],  # 6
                                              [9., 9., 1.],  # 7
                                              [2., 2., 2.]  # 8
                                              ])
        super(QuantumSuperiorityGame, self).__init__(num_players=num_players,
                                                     reward_distribution=reward_distribution)

    def _operator(self, *params) -> Operator:
        return self._general_local_operator(*params)
