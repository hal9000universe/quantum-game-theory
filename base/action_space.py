# py
from typing import List, Iterable
from math import pi

# nn & rl
from torch import Tensor, tensor, linspace, exp, cos, sin, cat

# lib
from base.quantum import Ops


"""This file is about action spaces of quantum games."""


class ActionSpace:
    """The ActionSpace class serves as a general template
    for implementing action spaces which can be integrated easily
    into the rest of the code. By subclassing and implementing
    the operator method, new actions spaces can be created.
    To integrate new action spaces into the quantum game
    reinforcement learning framework, the action space of
    any Env (base/env.py) can be adjusted via a setter method
    or otherwise specified in the definition of a new environment."""
    _ranges: Tensor
    _num_steps: int
    _num_params: int
    _ops: Ops
    _iterator: Iterable

    def __init__(self, ranges: Tensor):
        """Attributes
        self._ranges: torch.Tensor of lower and upper bounds for the parameters,
        e.g. tensor([[lower bound param 1, upper bound param 1],
                    [lower bound param 2, upper bound param 2]]).
        self._num_steps: int controlling the coarseness of discretization - the higher the finer.
        self._num_params: int specifying the number of parameters.
        self._ops: Ops object containing a variety of quantum operators.
        self._iterator: Iterable over all possible combinations of parameters."""
        self._ranges = ranges
        self._num_steps = 8
        self._num_params = len(ranges)
        self._ops = Ops()
        self._iterator = self._generate_iterator()

    @property
    def mins(self) -> Tensor:
        """The mins method returns the lower bounds
         of the action spaces' parameters."""
        return self._ranges[:, 0]

    @property
    def maxs(self) -> Tensor:
        """The maxs method returns the upper bounds
        of the action spaces' parameters."""
        return self._ranges[:, 1]

    def _generate_iterator(self) -> Iterable:
        """The _generate_iterator method creates an iterator
        which runs over all possible parameter combinations"""
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


class RestrictedActionSpace(ActionSpace):

    """The RestrictedActionSpace class implements the two-parameter action space
    described in quantum games and quantum strategies by Eisert et al."""

    def __init__(self):
        ranges: Tensor = tensor([[0., pi], [0., pi/2]])
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
