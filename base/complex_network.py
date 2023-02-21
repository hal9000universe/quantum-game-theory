# py
from math import pi

# nn & rl
from torch import Tensor, tensor, complex64, sigmoid, real, float32, relu
from torch.nn import Linear, Module
from torch.nn.init import kaiming_normal_


"""This file is about the ComplexNetwork architecture."""


class ComplexNetwork(Module):
    """The ComplexNetwork class implements a feed-forward neural network
    which is capable of handling 4-dimensional complex inputs."""
    _lin1: Linear
    _lin2: Linear
    _lin3: Linear
    _scaling: Tensor

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        self._lin1 = Linear(4, 128, dtype=complex64)
        self._lin2 = Linear(128, 128, dtype=float32)
        self._lin3 = Linear(128, 2, dtype=float32)
        self._scaling = tensor([pi, pi / 2])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        """
        Initializes weights using the kaiming-normal distribution and sets biases to zero.
        """
        if isinstance(m, Linear):
            m.weight = kaiming_normal_(m.weight)
            m.bias.data.fill_(0.)

    def __call__(self, x: Tensor, *args) -> Tensor:
        x = self._lin1(x)
        x = real(x)
        x = relu(x)
        x = self._lin2(x)
        x = relu(x)
        x = self._lin3(x)
        x = self._scaling * sigmoid(x)  # scales the output to fit the RestrictedActionSpace (base/action_space.py)
        return x
