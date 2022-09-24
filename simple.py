from torch import tensor, Tensor, relu, normal, full
from torch.nn import Module, Linear, Softplus
from torch.distributions import Normal, Uniform
from torch.optim import Adam
from statistics import mean
from typing import List
from random import uniform
from math import exp


class Env:
    _number: Tensor
    _rewards: List[float]
    _out: Tensor

    def __init__(self):
        self._number = tensor(1524.)
        self._rewards = list()
        self._out = tensor([1., 1., 1., 1.])

    def reset(self) -> Tensor:
        return self._out

    def step(self, guess: Tensor) -> Tensor:
        dif: Tensor = (self._number - guess).square()
        reward = tensor(100.) - dif
        self._rewards.append(reward.item())
        if len(self._rewards) > 50:
            self._rewards.pop(0)
        return reward

    def average_reward(self) -> float:
        return mean(self._rewards)


class Network(Module):

    def __init__(self):
        super(Network, self).__init__()
        self._lin1 = Linear(4, 64)
        self._lin1.weight.data = full((64, 4), 1.)
        self._lin2 = Linear(64, 32)
        self._lin2.weight.data = full((32, 64), 1.)
        self._lin3 = Linear(32, 2)
        self._lin3.weight.data = full((2, 32), 1.)
        self._soft_plus = Softplus()
        self._stabilizer = tensor([0., 1e-4])

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = relu(x)
        x = self._lin2(x)
        x = relu(x)
        x = self._lin3(x)
        x = self._soft_plus(x) + self._stabilizer
        return x


def sample(params: Tensor) -> Tensor:
    return normal(*params)


def log_prob(guess: Tensor, params: Tensor) -> Tensor:
    dist = Normal(*params)
    return dist.log_prob(guess)


class Agent:

    def __init__(self, env: Env, network: Network):
        self._env = env
        self._network = network
        self._optimizer = Adam(network.parameters())

    def train(self):
        rand_dist = Uniform(-2000., 2000.)
        for episode in range(10000000):
            state = self._env.reset()
            prob_params = self._network(state)
            if uniform(0., 1.) < exp(-episode):
                guess = rand_dist.sample()
            else:
                guess = sample(prob_params)
            reward = self._env.step(guess)
            loss: Tensor = -reward * log_prob(guess, prob_params)  # reward * log prob !
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if episode % 1000 == 0:
                print(f'{guess.item():.5}')


if __name__ == '__main__':
    envi = Env()
    netw = Network()
    agen = Agent(envi, netw)
    agen.train()


# TODO: stabilize learning across all scales (e.g. target = 1.)
# TODO: enable learning until convergence
