# nn & rl
from torch import Tensor, tensor, complex64, relu, kron, exp, sin, cos, normal
from torch.nn import Module, Linear, Softplus
from torch.optim import Adam

# py
from statistics import mean


class Network(Module):

    def __init__(self):
        super(Network, self).__init__()
        self._lin1 = Linear(8, 64)
        self._lin2 = Linear(64, 32)
        self._lin3 = Linear(32, 4)
        self._soft_plus = Softplus()

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = relu(x)
        x = self._lin2(x)
        x = relu(x)
        x = self._lin3(x)
        x = self._soft_plus(x)
        return x


def sample(params: Tensor) -> Tensor:
    return normal(*params)


if __name__ == '__main__':
    net = Network()
    optim = Adam(net.parameters(), lr=1e-5)
    label = tensor([1., 0., 0., 0.])
    inp = tensor([1.2, 3.4, 5.5, 3.4, 6.6, 4.7, 8.9, 9.3])
    losses = []
    for episode in range(100000):
        output = net(inp)
        loss: Tensor = (label - output).square().sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if len(losses) > 100:
            losses.pop(0)
        if episode % 100 == 0:
            print('Episode: {} - Loss: {}'.format(episode, mean(losses)))
