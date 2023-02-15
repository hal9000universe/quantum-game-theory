# py
from math import pi

# nn & rl
from torch import tensor, Tensor, complex64, eye, kron, int64, matrix_exp, exp, sin, cos, cat
from torch.distributions import Multinomial, Distribution
from torch.nn.functional import one_hot


def tens_pow(n: int, t: Tensor) -> Tensor:
    """The tens_pow function applies the tensor product to a Tensor t and itself n times."""
    out: Tensor = t
    for i in range(1, n):
        out = kron(out, t)
    return out


class QuantumSystem:
    """The QuantumSystem class implements a simulation of simple qubit systems."""
    _state: Tensor
    _num_qubits: int

    def __init__(self, num_qubits: int):
        """Attributes
        _state: a Tensor containing the coefficients of the quantum state.
        _num_qubits: int specifying the number of qubits in the system."""
        self._state = one_hot(tensor(0, dtype=int64), pow(2, num_qubits)).type(complex64)
        self._num_qubits = num_qubits

    def measure(self) -> Tensor:
        """The measure method simulates the measurement process of a qubit system."""
        probs: Tensor = self._state.abs().square()
        dist: Distribution = Multinomial(probs=probs)
        measurement: Tensor = dist.sample(1)
        return measurement

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def probs(self) -> Tensor:
        return self._state.abs().square()

    @property
    def state(self) -> Tensor:
        return self._state

    @state.setter
    def state(self, value: Tensor):
        self._state = value

    def __repr__(self) -> str:
        return f'{self._state}'


class Operator:
    """The Operator class serves as a template for quantum operators.
    Tensor products of Operators are implemented via the + operator.
    Applying an operator to a QuantumSystem is implemented via the @ operator."""
    _mat: Tensor

    def __init__(self, mat: Tensor):
        self._mat = mat

    def __matmul__(self, other: QuantumSystem) -> QuantumSystem:
        qs: QuantumSystem = QuantumSystem(num_qubits=other.num_qubits)
        qs.state = self.mat @ other.state
        return qs

    def __add__(self, other):
        return Operator(mat=kron(self.mat, other.mat))

    @property
    def mat(self) -> Tensor:
        return self._mat

    @property
    def adjoint(self):
        return Operator(self.mat.adjoint())

    def __repr__(self) -> str:
        return '{}'.format(self.mat)


class I(Operator):
    """The identity operator leaving a quantum state unchanged."""

    def __init__(self, dims: int = 2):
        iden: Tensor = eye(dims).type(complex64)
        super(I, self).__init__(mat=iden)

    @classmethod
    def inject(cls, dims: int = 2) -> Operator:
        return cls(dims=dims)


class PauliX(Operator):
    """The pauli_x operator switching the coefficients of a single qubit state."""

    def __init__(self):
        pauli_x: Tensor = tensor([[0., 1.],
                                  [1., 0.]], dtype=complex64)
        super(PauliX, self).__init__(mat=pauli_x)


class PauliY(Operator):
    """The pauli_y operator."""

    def __init__(self):
        pauli_y: Tensor = tensor([[0., -1j],
                                  [1j, 0.]], dtype=complex64)
        super(PauliY, self).__init__(mat=pauli_y)


class PauliZ(Operator):
    """The pauli_z operator."""

    def __init__(self):
        pauli_z: Tensor = tensor([[1., 0.],
                                  [0., -1.]], dtype=complex64)
        super(PauliZ, self).__init__(mat=pauli_z)


class RotX(Operator):
    """The x-rotation operator."""

    def __init__(self, rotx: Tensor = PauliX().mat):
        super(RotX, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliX().mat
        rotx: Tensor = matrix_exp(exponent)
        return cls(rotx=rotx)


class RotY(Operator):
    """The y-rotation operator."""

    def __init__(self, rotx: Tensor = PauliY().mat):
        super(RotY, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliY().mat
        roty: Tensor = matrix_exp(exponent)
        return cls(rotx=roty)


class RotZ(Operator):
    """The z-rotation operator."""

    def __init__(self, rotx: Tensor = PauliZ().mat):
        super(RotZ, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliZ().mat
        rotz: Tensor = matrix_exp(exponent)
        return cls(rotx=rotz)


class H(Operator):
    """The hadamard operator bringing a qubit into a uniform superposition."""

    def __init__(self):
        hada = tensor([[1., 1.],
                       [1., -1.]], dtype=complex64) / tensor(2.).sqrt()
        super(H, self).__init__(mat=hada)


class CNOT(Operator):
    """The CNOT operator maximally entangling two qubits,
    after the hadamard operator is applied to the control qubit."""

    def __init__(self):
        cnot: Tensor = tensor([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 1., 0.]], dtype=complex64)
        super(CNOT, self).__init__(mat=cnot)


class GeneralJ(Operator):
    """The J Operator maximally entangling an arbitrary number of qubits as described in Multi-Player Quantum Games."""

    def __init__(self, num_players: int = 2):
        prep: Tensor = 1 / tensor(2.).sqrt() * (
                tens_pow(num_players, I().mat) + 1j * tens_pow(num_players, PauliX().mat))
        super(GeneralJ, self).__init__(mat=prep)

    @classmethod
    def inject(cls, num_players: int) -> Operator:
        return cls(num_players=num_players)


class J(Operator):
    """The J Operator maximally entangling two qubits as described
    in Quantum Games and Quantum Strategies by Eisert et al."""

    def __init__(self, gamma: float = pi / 2):
        D: Tensor = tensor([[0., 1.],
                            [-1., 0.]], dtype=complex64)
        J_mat = matrix_exp(-1j * gamma * kron(D, D) / 2)
        super(J, self).__init__(mat=J_mat)

    @classmethod
    def inject(cls, gamma: float) -> Operator:
        return cls(gamma)


class Ops:
    """A collection of quantum operators."""
    _I: I
    _sx: PauliX
    _sy: PauliY
    _sz: PauliZ
    _H: H
    _CNOT: CNOT
    _J: GeneralJ
    _RotX: RotX
    _RotY: RotY
    _RotZ: RotZ

    def __init__(self):
        self._I = I()
        self._sx = PauliX()
        self._sy = PauliY()
        self._sz = PauliZ()
        self._H = H()
        self._CNOT = CNOT()
        self._GeneralJ = GeneralJ()
        self._J = J()
        self._RotX = RotX()
        self._RotY = RotY()
        self._RotZ = RotZ()

    @property
    def I(self) -> I:
        return self._I

    @property
    def sx(self) -> PauliX:
        return self._sx

    @property
    def sy(self) -> PauliY:
        return self._sy

    @property
    def sz(self) -> PauliZ:
        return self._sz

    @property
    def H(self) -> H:
        return self._H

    @property
    def CNOT(self) -> CNOT:
        return self._CNOT

    @property
    def GeneralJ(self) -> GeneralJ:
        return self._GeneralJ

    @property
    def J(self):
        return self._J

    @property
    def RotX(self) -> RotX:
        return self._RotX

    @property
    def RotY(self) -> RotY:
        return self._RotY

    @property
    def RotZ(self) -> RotZ:
        return self._RotZ
