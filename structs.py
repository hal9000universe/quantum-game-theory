# py
from typing import Tuple, List

# nn & rl
from torch import exp, cos, sin, square, sqrt, kron, tensor, Tensor, abs, complex64, concat, real, imag, eye


def rotation_matrix(theta: Tensor, phi: Tensor) -> Tensor:
    # defines a rotation matrix and returns it
    a = exp(1j * phi) * cos(theta / 2)
    b = sin(theta / 2)
    c = - sin(theta / 2)
    d = exp(-1j * phi) * cos(theta / 2)
    return tensor([[a, b], [c, d]])


class State:
    _representation: Tensor
    _bases: List[Tuple[bool, bool]]

    def __init__(self):
        self._representation = tensor([1., 0, 0, 1.], dtype=complex64) / sqrt(tensor(2., dtype=complex64))
        self._bases = [(False, False), (False, True), (True, False), (True, True)]

    def measure(self) -> int:
        probs: ndarray = self._representation.abs().square()
        measurement: int = probs.multinomial(num_samples=1, replacement=True)
        self._reset()
        return measurement

    def long(self) -> Tensor:
        return concat((real(self._representation), imag(self._representation)))

    def _reset(self):
        self._representation = tensor([1., 0, 0, 1.], dtype=complex64) / sqrt(tensor(2, dtype=complex64))

    @property
    def representation(self) -> Tensor:
        return self._representation

    @representation.setter
    def representation(self, state: Tensor):
        self._representation = self._normalize(state)

    @staticmethod
    def _normalize(state: Tensor) -> Tensor:
        norm = state.abs().sum()
        if norm == 1.:
            return state
        else:
            return state / norm

    def __repr__(self) -> str:
        a, b, c, d = self._representation
        return f'{a:.2}|00> + {b:.2}|01> + {c:.2}|10> + {d:.2}|11>'


class Operator:

    def __init__(self):
        self.hadamard = tensor([[1., 1.], [1., -1.]], dtype=complex64) / sqrt(tensor(2.))
        self.cnot = tensor([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 0., 1.],
                            [0., 0., 1., 0.]], dtype=complex64)
        self.identity = eye(2, dtype=complex64)

    def cnot(self, multi_qubit_system: Tensor):
        return self.cnot @ multi_qubit_system

    @staticmethod
    def rotation_matrix(theta: Tensor, phi: Tensor) -> Tensor:
        # defines a rotation matrix and returns it
        a = exp(1j * phi) * cos(theta / 2)
        b = sin(theta / 2)
        c = - sin(theta / 2)
        d = exp(-1j * phi) * cos(theta / 2)
        return tensor([[a, b], [c, d]])

    def rotate_qubits(self, theta_a: Tensor, phi_a: Tensor, theta_b: Tensor, phi_b: Tensor, state: Tensor) -> Tensor:
        alice_rot: Tensor = self.rotation_matrix(theta_a, phi_a)
        bob_rot: Tensor = self.rotation_matrix(theta_b, phi_b)
        matrix_representation = kron(alice_rot, bob_rot)
        return matrix_representation @ state
