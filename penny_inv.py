from pennylane import Device, device, qnode, RZ, RX, QubitUnitary, probs
from quantum import Ops, QuantumSystem, Operator
from torch import Tensor, tensor
from math import pi


@qnode(device=device('default.qubit',  wires=2), interface='torch')
def quantum_circuit(rots1: Tensor, rots2: Tensor) -> Tensor:
    QubitUnitary(Ops().J.inject(2).mat, wires=[0, 1])
    RZ(rots1[0], wires=0)
    RX(rots1[1], wires=0)
    RZ(rots1[2], wires=0)
    RZ(rots2[0], wires=1)
    RX(rots2[1], wires=1)
    RZ(rots2[2], wires=1)
    QubitUnitary(Ops().J.inject(2).mat.adjoint(), wires=[0, 1])
    return probs(wires=[0, 1])


@qnode(device=device('default.qubit', wires=2), interface='torch')
def optional_quantum_circuit(rots1: Tensor, rots2: Tensor) -> Tensor:
    QubitUnitary(Ops().J.inject(2).mat, wires=[0, 1])
    rotop1 = Operator(Ops().RotZ.inj(rots1[2]).mat @ Ops().RotX.inject(rots1[1]).mat @ Ops().RotZ.inj(rots1[0]).mat)
    rotop2 = Operator(Ops().RotZ.inj(rots2[2]).mat @ Ops().RotX.inject(rots2[1]).mat @ Ops().RotZ.inj(rots2[0]).mat)
    op = rotop1 + rotop2
    QubitUnitary(op.mat, wires=[0, 1])
    QubitUnitary(Ops().J.inject(2).mat.adjoint(), wires=[0, 1])
    return probs(wires=[0, 1])


def classical_circuit(rots1: Tensor, rots2: Tensor) -> Tensor:
    state = QuantumSystem(num_qubits=2)
    state = Ops().J @ state
    rotop1 = Operator(Ops().RotZ.inj(rots1[2]).mat @ Ops().RotX.inject(rots1[1]).mat @ Ops().RotZ.inj(rots1[0]).mat)
    rotop2 = Operator(Ops().RotZ.inj(rots2[2]).mat @ Ops().RotX.inject(rots2[1]).mat @ Ops().RotZ.inj(rots2[0]).mat)
    op = rotop1 + rotop2
    state = op @ state
    state = Operator(Ops().J.mat.adjoint()) @ state
    return state.probs


if __name__ == '__main__':
    params1 = tensor([0., pi, 0.])
    params2 = tensor([pi, 0., pi])
    print(optional_quantum_circuit(params1, params2))
    print(classical_circuit(params1, params2))
