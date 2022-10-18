# py
from typing import Optional, List

# nn & rl
from torch import Tensor, tensor

# lib
from quantum import Ops, Operator, QuantumSystem


class Env:
    _num_players: int
    _reward_distribution: Tensor
    _ops: Ops
    _J: Operator
    _state: QuantumSystem

    def __init__(self, num_players: int,
                 reward_distribution: Tensor):
        self._num_players = num_players
        self._reward_distribution = reward_distribution
        self._ops = Ops()
        self._J = self._ops.J.inject(num_players=num_players)
        self._state = QuantumSystem(num_qubits=num_players)

    def reset(self):
        self._state = QuantumSystem(num_qubits=self._num_players)
        self._state = self._J @ self._state
        return self._state

    def operator(self, *params) -> Operator:
        """
        returns an operator specified by
        """
        raise NotImplementedError

    def general_local_operator(self, theta1: Tensor, theta2: Tensor, theta3: Tensor) -> Operator:
        """
        returns a general local unitary operator specified by theta1, theta2 & theta3.
        """
        rot1 = self._ops.RotZ.inject(theta1).mat
        rot2 = self._ops.RotX.inject(theta2).mat
        rot3 = self._ops.RotZ.inject(theta3).mat
        op = rot3 @ rot2 @ rot1
        return Operator(mat=op)

    def step(self, *args) -> List[Tensor]:
        assert len(args) == self._num_players
        # create operator which applies the local unitary actions
        op: Tensor = operator(*args[0])
        for i in range(1, len(args)):
            op = kron(op, operator(*args[i]))
        # apply operator
        self._state = op @ self._state
        # apply adjoint state preparation operator
        self._state = self._J.adjoint() @ self._state
        # get probs
        probs = self._state.probs
        # get q-values
        qs: List = []
        for i in range(self._num_players):
            q_i = (self._reward_distribution[:, i] * probs).sum()
            qs.append(q_i)
        return qs
