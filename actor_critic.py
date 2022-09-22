# py
from typing import Callable, Tuple
from random import uniform
from math import pi

# nn & rl
from torch import Tensor, relu, tensor, normal, ones, kron
from torch.nn import Linear, Module, Softplus
from torch.distributions import Normal, Uniform
from torch.optim import Optimizer, Adam
import copy

# lib
from env import Env
from structs import State, rotation_matrix
from replay_buffer import ReplayBuffer, sample_batch


def log_prob(params: Tensor, action: Tensor) -> Tensor:
    mean = params[:, 0:2]
    std: Tensor = params[:, 2:4]
    distribution = Normal(mean, std)
    return distribution.log_prob(action)


def sample(params: Tensor) -> Tensor:
    mean: Tensor = params[0:2]
    std: Tensor = params[2:4]
    action: Tensor = normal(mean, std)
    return action


def compute_loss(par: Tensor, ac: Tensor, delta: Tensor) -> Tensor:
    loss = (log_prob(par, ac).transpose(0, 1) * delta).transpose(0, 1).sum(axis=1).mean(axis=0)
    return loss


class Actor(Module):

    def __init__(self):
        super(Actor, self).__init__()
        self._lin1 = Linear(8, 4)
        self._soft_plus = Softplus()

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = self._soft_plus(x)
        return x


class Critic(Module):
    _lin1: Linear

    def __init__(self):
        super(Critic, self).__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return tensor(3.) * ones(64)  # 3. is the reward if both agents behave optimally


class AliceBobCritic:
    _alice: Module
    _bob: Module
    _critic: Module
    _env: Env
    _replay_buffer: ReplayBuffer
    _alice_optimizer: Optimizer
    _bob_optimizer: Optimizer
    _critic_optimizer: Optimizer

    def __init__(self,
                 alice: Module,
                 bob: Module,
                 critic: Module,
                 env: Env,
                 replay_buffer: ReplayBuffer,
                 alice_optimizer: Optimizer,
                 bob_optimizer: Optimizer,
                 # critic_optimizer: Optimizer,
                 ):
        self._alice = alice
        self._bob = bob
        self._critic = critic
        self._env = env
        self._replay_buffer = replay_buffer
        self._alice_optimizer = alice_optimizer
        self._bob_optimizer = bob_optimizer
        self._epsilon = 0.3
        self._param_dist = Uniform(0., 2*pi)
        # self._critic_optimizer = critic_optimizer

    def _choose_actions(self, state: State) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        state_representation = state.long()  # save state.long() before operator is applied
        if uniform(0, 1) < self._epsilon:
            theta_a, phi_a, theta_b, phi_b = self._param_dist.sample((4,))
        else:
            theta_a, phi_a = sample(self._alice(state_representation))
            theta_b, phi_b = sample(self._alice(state_representation))
        return theta_a, phi_a, theta_b, phi_b

    def run(self):
        for episode in range(100000):
            state: State = self._env.reset()
            theta_a, phi_a, theta_b, phi_b = self._choose_actions(state)
            self._env.step(theta_a, phi_a, theta_b, phi_b, state.long())
            states, actions_alice, actions_bob, reward_alice, reward_bob = sample_batch(
                self._replay_buffer.size,
                self._replay_buffer.states,
                self._replay_buffer.actions_alice,
                self._replay_buffer.actions_bob,
                self._replay_buffer.rewards_alice,
                self._replay_buffer.rewards_bob,
                64,
            )
            delta_alice: Tensor = reward_alice - self._critic(states)
            delta_bob: Tensor = reward_bob - self._critic(states)
            # loss_critic: Tensor = delta_alice.square().mean() + delta_bob.square().mean()
            # self._critic_optimizer.zero_grad()
            # loss_critic.backward()
            # self._critic_optimizer.step()

            loss_alice: Tensor = compute_loss(self._alice(states), actions_alice, delta_alice)
            loss_bob: Tensor = compute_loss(self._alice(states), actions_bob, delta_bob)

            self._alice_optimizer.zero_grad()
            # self._bob_optimizer.zero_grad()

            loss_alice.backward()
            loss_bob.backward()

            self._alice_optimizer.step()
            # self._bob_optimizer.step()

            if episode % 100 == 0:
                print('average: {}'.format(self._env.average_reward))


if __name__ == '__main__':
    a = Actor()
    b = Actor()
    c = Critic()

    buffer_size = 100000
    rep_buf = ReplayBuffer(
        buffer_size=buffer_size,
        obs_shape=(buffer_size, 8),
        ac_shape=(buffer_size, 2),
    )

    environment = Env(rep_buf)

    a_optim = Adam(a.parameters())
    b_optim = Adam(b.parameters())
    # c_optim = Adam(c.parameters())

    abc = AliceBobCritic(a, b, c, environment, rep_buf, a_optim, b_optim)
    # abc.run()

    identity = tensor([[1., 0.], [0., 1.]])
    rot = rotation_matrix(tensor(pi), tensor(1.3))
    s = State()
    print(f"normal state: {s}")
    s1 = State()
    s1.representation = kron(identity, rot) @ (kron(rot, identity) @ s1.representation)
    print(f"alice then bob state: {s1}")
    s2 = State()
    s2.representation = kron(rot, identity) @ (kron(identity, rot) @ s2.representation)
    print(f"bob then alice state: {s2}")
    s3 = State()
    s3.representation = kron(rot, rot) @ s3.representation
    print(f"simultaneously alice and bob state: {s3}")
