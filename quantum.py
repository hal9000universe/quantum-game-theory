from torch import tensor, Tensor, zeros, complex64, eye, sqrt, kron, int64
from torch.distributions import Multinomial, Distribution
from torch.nn.functional import one_hot


def tens_pow(n: int, t: Tensor) -> Tensor:
    out: Tensor = t
    for i in range(n):
        out = kron(out, t)
    return out


class QuantumSystem:
    _state: Tensor

    def __init__(self, num_qubits: int):
        self._state = one_hot(tensor(0, dtype=int64), pow(2, num_qubits)).type(complex64)

    def measure(self) -> Tensor:
        probs: Tensor = self._state.abs().square()
        dist: Distribution = Multinomial(probs=probs)
        measurement: Tensor = dist.sample(1)
        return measurement

    @property
    def probs(self) -> Tensor:
        return self._state.abs().square()

    @property
    def state(self) -> Tensor:
        return self._state

    @state.setter
    def state(self, value: Tensor):
        self._state = value


class Operator:
    _mat: Tensor

    def __init__(self, mat: Tensor):
        self._mat = mat

    @property
    def mat(self) -> Tensor:
        return self._mat


class I(Operator):

    def __init__(self, dims: int = 2):
        iden: Tensor = eye(dims)
        super(I, self).__init__(mat=iden)

    @classmethod
    def inject(cls, dims: int = 2) -> Operator:
        return cls(dims=dims)


class F(Operator):
    def __init__(self):
        flip: Tensor = tensor([[]])
        super(F, self).__init__(mat=flip)


class H(Operator):

    def __init__(self):
        hada = tensor([[1., 1.],
                       [1., -1.]], dtype=complex64) / tensor(2.).sqrt()
        super(H, self).__init__(mat=hada)


class CNOT(Operator):

    def __init__(self):
        cnot = tensor([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 1., 0.]], dtype=complex64)
        super(CNOT, self).__init__(mat=cnot)


class J(Operator):

    def __init__(self, num_players: int = 2):
        prep: Tensor = 1 / tensor(2.).sqrt() * (tens_pow(num_players, I().mat) + 1j * tens_pow(num_players, F().mat))
        super(J, self).__init__(mat=prep)

    @classmethod
    def inject(cls, num_players: int) -> Operator:
        return cls(num_players=num_players)


class Ops:
    _I: I
    _F: F
    _H: H
    _CNOT: CNOT
    _J: J

    def __init__(self):
        self._I = I()
        self._F = F()
        self._H = H()
        self._CNOT = CNOT()
        self._J = J()

    @property
    def I(self) -> I:
        return self._I

    @property
    def F(self) -> F:
        return self._F

    @property
    def H(self) -> H:
        return self._H

    @property
    def CNOT(self) -> CNOT:
        return self._CNOT

    @property
    def J(self) -> J:
        return self._J


if __name__ == '__main__':
    sys = QuantumSystem(num_qubits=2)
