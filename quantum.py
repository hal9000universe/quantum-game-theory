from torch import Tensor, tensor, complex64, sqrt, eye, kron, conj
from torch.nn.functional import one_hot


def adjoint(matrix: Tensor) -> Tensor:
    return conj(matrix.transpose(0, 1))


def commute(mat1: Tensor, mat2: Tensor) -> bool:
    commutator = mat1 @ mat2 - mat2 @ mat1
    if commutator.sum().abs() < 1e-8:
        return True
    else:
        return False


class Operator:
    _matrix_representation: Tensor

    def __init__(self, matrix_representation: Tensor):
        self._matrix_representation = matrix_representation

    @property
    def matrix_representation(self) -> Tensor:
        return self._matrix_representation

    def apply(self, state: Tensor) -> Tensor:
        return self._matrix_representation @ state

    def is_unitary(self) -> bool:
        identity = eye(self._matrix_representation.size(dim=0))
        dif = identity - self._matrix_representation @ adjoint(self._matrix_representation)
        if dif.abs().square().sum() < 1e-8:
            return True
        else:
            return False


class Identity(Operator):

    def __init__(self):
        matrix_representation = eye(2)
        super(Identity, self).__init__(matrix_representation=matrix_representation)


class Hadamard(Operator):

    def __init__(self):
        matrix_representation = tensor([[1., 1.],
                                        [1., -1.]], dtype=complex64) / sqrt(tensor(2.))
        super(Hadamard, self).__init__(matrix_representation=matrix_representation)


class CNOT(Operator):

    def __init__(self):
        matrix_representation = tensor([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 0., 1.],
                                        [0., 0., 1., 0.]], dtype=complex64)
        super(CNOT, self).__init__(matrix_representation=matrix_representation)


class Rotation(Operator):

    def __init__(self):
        matrix_representation = eye(2, dtype=complex64)
        super(Rotation, self).__init__(matrix_representation=matrix_representation)

    def inject(self, theta: Tensor, phi: Tensor, state: Tensor) -> Tensor:
        self._matrix_representation = tensor([[exp(1j * phi) * cos(theta / 2), sin(theta / 2)],
                                              [-sin(theta / 2), exp(-1j * phi) * cos(theta / 2)]], dtype=complex64)
        return self.apply(state)


class General(Operator):

    def __init__(self):
        matrix_representation = eye(2, dtype=complex64)
        super(General, self).__init__(matrix_representation=matrix_representation)

    def inject(self, matrix_representation: Tensor, state: Tensor) -> Tensor:
        self._matrix_representation = matrix_representation
        assert self.is_unitary()
        return self.apply(state)


class Cooperate(Operator):

    def __init__(self):
        matrix_representation = eye(2, dtype=complex64)
        super(Cooperate, self).__init__(matrix_representation=matrix_representation)


class Defect(Operator):

    def __init__(self):
        matrix_representation = tensor([[0., 1.],
                                        [1., 0.]], dtype=complex64)
        super(Defect, self).__init__(matrix_representation=matrix_representation)


class Preparation(Operator):

    def __init__(self):
        matrix_representation = CNOT().matrix_representation @ kron(Hadamard().matrix_representation,
                                                                    Identity().matrix_representation)
        super(Preparation, self).__init__(matrix_representation=matrix_representation)


class Ops:
    identity: Operator
    hadamard: Operator
    cnot: Operator
    rotation: Operator
    general: Operator
    cooperate: Operator
    defect: Operator
    preparation: Operator

    def __init__(self):
        self.identity = Identity()
        self.hadamard = Hadamard()
        self.cnot = CNOT()
        self.rotation = Rotation()
        self.general = General()
        self.cooperate = Cooperate()
        self.defect = Defect()
        self.preparation = Preparation()


def normalize(state: Tensor) -> Tensor:
    norm = state.abs().sum()
    if norm == 1.:
        return state
    else:
        return state / norm


class QuantumSystem:
    _num_qubits: int
    _state: Tensor

    def __init__(self, num_qubits: int = 2):
        self._num_qubits = num_qubits
        self._state = one_hot(tensor(0), pow(2, num_qubits)).type(complex64)

    @property
    def state(self) -> Tensor:
        return self._state

    @state.setter
    def state(self, state: Tensor):
        assert state.size(dim=0) == pow(2, self._num_qubits)
        self._state = normalize(state)

    def reset(self):
        self._state = one_hot(tensor(0), classes=pow(2, self._num_qubits))

    def measure(self) -> int:
        probs: Tensor = self._state.abs().square()
        measurement: int = probs.multinomial(num_samples=1, replacement=True)
        self._reset()
        return measurement


class TwoQubitSystem(QuantumSystem):
    _ops: Ops

    def __init__(self):
        super(TwoQubitSystem, self).__init__(num_qubits=2)
        self._ops = Ops()

    def prepare_state(self):
        self._state = self._ops.preparation.apply(self._state)

    def check_conditions(self) -> bool:
        def_def_mat = kron(self._ops.defect.matrix_representation, self._ops.defect.matrix_representation)
        prep_def_def_commute = commute(self._ops.preparation.matrix_representation, def_def_mat)
        def_coop_mat = kron(self._ops.defect.matrix_representation, self._ops.cooperate.matrix_representation)
        prep_def_coop_commute = commute(self._ops.preparation.matrix_representation, def_coop_mat)
        coop_def_mat = kron(self._ops.cooperate.matrix_representation, self._ops.defect.matrix_representation)
        prep_coop_def_commute = commute(self._ops.preparation.matrix_representation, coop_def_mat)
        if prep_def_def_commute and prep_def_coop_commute and prep_coop_def_commute:
            return True
        else:
            return False

    def __repr__(self) -> str:
        a, b, c, d = self._state
        return f'{a:.2}|00> + {b:.2}|01> + {c:.2}|10> + {d:.2}|11>'


if __name__ == '__main__':
    system = TwoQubitSystem()
    print(system.check_conditions())
