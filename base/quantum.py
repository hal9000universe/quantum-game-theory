# py
from math import pi

# nn & rl
from torch import tensor, Tensor, complex64, eye, kron, int64, matrix_exp, exp, sin, cos, cat
from torch.distributions import Multinomial, Distribution
from torch.nn.functional import one_hot


def tens_pow(n: int, t: Tensor) -> Tensor:
    out: Tensor = t
    for i in range(1, n):
        out = kron(out, t)
    return out


class QuantumSystem:
    _state: Tensor
    _num_qubits: int

    def __init__(self, num_qubits: int):
        self._state = one_hot(tensor(0, dtype=int64), pow(2, num_qubits)).type(complex64)
        self._num_qubits = num_qubits

    def measure(self) -> Tensor:
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

    def __init__(self, dims: int = 2):
        iden: Tensor = eye(dims).type(complex64)
        super(I, self).__init__(mat=iden)

    @classmethod
    def inject(cls, dims: int = 2) -> Operator:
        return cls(dims=dims)


class PauliX(Operator):

    def __init__(self):
        pauli_x: Tensor = tensor([[0., 1.],
                                  [1., 0.]], dtype=complex64)
        super(PauliX, self).__init__(mat=pauli_x)


class PauliY(Operator):

    def __init__(self):
        pauli_y: Tensor = tensor([[0., -1j],
                                  [1j, 0.]], dtype=complex64)
        super(PauliY, self).__init__(mat=pauli_y)


class PauliZ(Operator):

    def __init__(self):
        pauli_z: Tensor = tensor([[1., 0.],
                                  [0., -1.]], dtype=complex64)
        super(PauliZ, self).__init__(mat=pauli_z)


class RotX(Operator):

    def __init__(self, rotx: Tensor = PauliX().mat):
        super(RotX, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliX().mat
        rotx: Tensor = matrix_exp(exponent)
        return cls(rotx=rotx)


class RotY(Operator):

    def __init__(self, rotx: Tensor = PauliY().mat):
        super(RotY, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliY().mat
        roty: Tensor = matrix_exp(exponent)
        return cls(rotx=roty)


class RotZ(Operator):

    def __init__(self, rotx: Tensor = PauliZ().mat):
        super(RotZ, self).__init__(mat=rotx)

    @classmethod
    def inject(cls, theta: Tensor) -> Operator:
        exponent: Tensor = 1j / 2 * theta * PauliZ().mat
        rotz: Tensor = matrix_exp(exponent)
        return cls(rotx=rotz)


class H(Operator):

    def __init__(self):
        hada = tensor([[1., 1.],
                       [1., -1.]], dtype=complex64) / tensor(2.).sqrt()
        super(H, self).__init__(mat=hada)


class CNOT(Operator):

    def __init__(self):
        cnot: Tensor = tensor([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 1., 0.]], dtype=complex64)
        super(CNOT, self).__init__(mat=cnot)


class GeneralJ(Operator):

    def __init__(self, num_players: int = 2):
        prep: Tensor = 1 / tensor(2.).sqrt() * (
                tens_pow(num_players, I().mat) + 1j * tens_pow(num_players, PauliX().mat))
        super(GeneralJ, self).__init__(mat=prep)

    @classmethod
    def inject(cls, num_players: int) -> Operator:
        return cls(num_players=num_players)


class J(Operator):

    def __init__(self, gamma: float = pi / 2):
        D: Tensor = tensor([[0., 1.],
                            [-1., 0.]], dtype=complex64)
        J_mat = matrix_exp(-1j * gamma * kron(D, D) / 2)
        super(J, self).__init__(mat=J_mat)

    @classmethod
    def inject(cls, gamma: float) -> Operator:
        return cls(gamma)


class U(Operator):

    def __init__(self):
        mat: Tensor = eye(2).type(complex64)
        super(U, self).__init__(mat=mat)

    @classmethod
    def inject(cls, theta: Tensor, phi: Tensor, psi: Tensor) -> Operator:
        a: Tensor = cos(theta / 2)
        b: Tensor = sin(theta / 2)
        c: Tensor = -exp(1j * psi)
        d: Tensor = exp(1j * phi)
        e: Tensor = exp(1j * (phi + psi))
        # construct the rows of the rotation matrix
        r1: Tensor = cat((a.view(1), (b * c).view(1)))
        r2: Tensor = cat(((b * d).view(1), (a * e).view(1)))
        # build and return the rotation matrix
        rot: Tensor = cat((r1, r2)).view(2, 2)
        return Operator(mat=rot)


class Ops:
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
    _U: U

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
        self._U = U()

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

    @property
    def U(self) -> U:
        return self._U
