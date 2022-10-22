from pennylane import Device, device, qnode, QubitUnitary, probs
from quantum import QuantumSystem, Operator, Ops
from torch import Tensor, tensor, kron, complex64, matrix_exp
from math import pi
from typing import Tuple

O: Ops = Ops()
dev: Device = device('default.qubit', wires=2)

gamma: float = pi / 2
D: Tensor = tensor([[0., 1.],
                    [-1., 0.]], dtype=complex64)
mat = matrix_exp(-1j * gamma * kron(D, D) / 2)
J = Operator(mat)


@qnode(device=dev, interface='torch')
def quantum_circuit(op1: Tensor, op2: Tensor) -> Tensor:
    QubitUnitary(J.mat, wires=[0, 1])
    QubitUnitary(op1, wires=0)
    QubitUnitary(op2, wires=1)
    QubitUnitary(J.mat.adjoint(), wires=[0, 1])
    return probs(wires=[0, 1])


def classical_circuit(op1: Tensor, op2: Tensor) -> Tensor:
    state: QuantumSystem = QuantumSystem(num_qubits=2)
    state = J @ state
    state = Operator(kron(op1, op2)) @ state
    state = J.adjoint @ state
    return state.probs


if __name__ == '__main__':
    operation1: Tensor = tensor([[0., 1.],
                                 [-1., 0.]], dtype=complex64)
    operation2: Tensor = tensor([[1j, 0.],
                                 [0., -1j]], dtype=complex64)

    q_probs: Tensor = quantum_circuit(operation1, operation2)
    c_probs: Tensor = classical_circuit(operation1, operation2)

    print(q_probs)
    print(c_probs)

    rewards: Tensor = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

    qq: Tuple[Tensor, Tensor] = ((q_probs * rewards[:, 0]).sum(), (q_probs * rewards[:, 1]).sum())
    cq: Tuple[Tensor, Tensor] = ((c_probs * rewards[:, 0]).sum(), (c_probs * rewards[:, 1]).sum())

    print(qq)
    print(cq)
