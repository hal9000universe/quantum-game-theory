# py
from abc import ABC
from typing import Callable, List, Optional, Tuple
from math import sqrt

# qml
from pennylane import device, PauliX, Identity, matrix, RX, RZ, expval, qnode, QubitUnitary
from pennylane.operation import Operation, AnyWires
from torch import Tensor

# lib
from quantum import Ops


def create_circuit(num_players: int) -> Callable:
    # creating quantum device
    dev: device = device('qiskit.aer', wires=num_players)

    @qnode(device=dev, interface='torch')
    def circuit(*args: Tuple[Tensor, ...]) -> List[float]:
        # prepare state
        QubitUnitary(Ops().J.inject(num_players=num_players).mat, wires=[i for i in range(num_players)])
        # apply rotations specified by args
        for wire in range(num_players):
            theta1, theta2, theta3 = args[wire]
            RZ(theta1, wires=[wire])
            RX(theta2, wires=[wire])
            RZ(theta3, wires=[wire])
        # prepare final state
        QubitUnitary(Ops().J.inject(num_players=num_players).mat.adjoint(), wires=[i for i in range(num_players)])
        # calculate expectation value
        return [expval(PauliX(wires=wire)) for wire in range(num_players)]
    return circuit


if __name__ == '__main__':
    circ = create_circuit(num_players=3)
    print(circ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)))
