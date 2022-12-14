# py
from typing import List, Tuple, Iterable
from math import pi

# nn & rl
from torch import Tensor, linspace, exp, cos, sin, cat

# lib
from base.quantum import Ops


class ActionSpace:
    _ranges: List[Tuple[float, float]]
    _num_steps: int
    _num_params: int
    _ops: Ops
    _iterator: Iterable

    def __init__(self, ranges: List[Tuple[float, float]]):
        self._ranges = ranges
        self._num_steps = 8
        self._num_params = len(ranges)
        self._ops = Ops()
        self._iterator = self._generate_iterator()

    def _generate_iterator(self) -> Iterable:
        params: List[Tensor] = []
        for angle_range in self._ranges:
            lin_space: Tensor = linspace(angle_range[0], angle_range[1], steps=self._num_steps)
            params.append(lin_space)
        power: int
        dims: int
        for i in range(0, self._num_params):
            power = len(self._ranges) - 1 - i
            dims = i
            params[i] = params[i].repeat_interleave(self._num_steps ** power)
            params[i] = params[i].broadcast_to((self._num_steps ** dims, self._num_steps ** (power + 1))).flatten()
        return zip(*params)

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int):
        self._num_steps = num_steps

    @property
    def iterator(self) -> Iterable:
        self._iterator = self._generate_iterator()
        return self._iterator

    def operator(self, params: Tensor) -> Tensor:
        raise NotImplementedError


class GeneralActionSpace(ActionSpace):

    def __init__(self):
        # euler angle ranges: [-pi, pi], [0, pi], [-pi, pi]
        ranges: List[Tuple[float, float]] = [(-pi, pi), (0., pi), (-pi, pi)]
        super(GeneralActionSpace, self).__init__(ranges=ranges)

    def operator(self, params: Tensor) -> Tensor:
        """
        computes and returns a complex rotation matrix given by the Euler angles in params.
        """
        return self._ops.U.inj(*params).mat


class RestrictedActionSpace(ActionSpace):

    def __init__(self):
        # angle ranges given in Quantum Games and Quantum Strategies by Eisert et al.
        ranges: List[Tuple[float, float]] = [(0., pi), (0., pi / 2)]
        super(RestrictedActionSpace, self).__init__(ranges=ranges)

    def operator(self, params: Tensor) -> Tensor:
        """
        computes and returns a complex rotation matrix given by the angles in params.
        """
        theta, phi = params
        # calculate entries
        a: Tensor = exp(1j * phi) * cos(theta / 2)
        b: Tensor = sin(theta / 2)
        c: Tensor = -b
        d: Tensor = exp(-1j * phi) * cos(theta / 2)
        # construct the rows of the rotation matrix
        r1: Tensor = cat((a.view(1), b.view(1)))
        r2: Tensor = cat((c.view(1), d.view(1)))
        # build and return the rotation matrix
        rot: Tensor = cat((r1, r2)).view(2, 2)
        return rot
