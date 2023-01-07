# py
from typing import List
from math import pi

# nn & rl
from torch import Tensor, tensor, complex64, sigmoid, real, float32, eye
from torch.nn import Linear, Module
from torch.nn.init import kaiming_normal_
from torch.optim import Optimizer, Adam

# lib
from base.env import Env
from base.general import train
from base.utils import create_env, calc_dist


class ComplexNetwork(Module):
    _lin1: Linear
    _lin2: Linear
    _lin3: Linear
    _scaling: Tensor

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        """
        initializes ComplexNetwork class.
         ComplexNetwork implements a feed-forward neural network which is capable of handling 
         4-dimensional complex inputs. 
        """
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
        x = self._lin2(x)
        x = self._lin3(x)
        x = self._scaling * sigmoid(x)
        return x


def test_complex_network(fix_inp: bool = True, noisy_actions: bool = False) -> float:
    num_successes: int = 0
    num_attempts: int = 100
    for attempt in range(0, num_attempts):
        env: Env = create_env()
        num_players: int = 2
        agents: List[ComplexNetwork] = [ComplexNetwork() for _ in range(0, num_players)]
        optimizers: List[Optimizer] = [Adam(params=agents[i].parameters()) for i in range(0, num_players)]
        episodes: int = 3000
        agents: List[ComplexNetwork] = train(
            episodes=episodes,
            fix_inp=fix_inp,
            fix_inp_time=int(episodes * 0.6),
            noisy_actions=noisy_actions,
            fix_actions_time=int(episodes * 0.6),
            num_players=num_players,
            agents=agents,
            optimizers=optimizers,
            env=env,
            player_tokens=eye(2),
            reward_distribution=env.reward_distribution
        )

        nash_eq: Tensor = tensor([[0., pi / 2], [0., pi / 2]])
        actions: Tensor = tensor([tuple(player(env.reset(fix_inp=True))) for player in agents])
        dist: Tensor = calc_dist(nash_eq, actions, env.action_space.operator)
        if dist <= 0.2:
            print(f"{attempt}: successful - {dist} - {actions}")
            num_successes += 1
        else:
            print(f"{attempt}: failed - {dist} - {actions}")

    success_rate: float = num_successes / num_attempts
    print(f"Performance: {success_rate}")

    return success_rate


if __name__ == '__main__':
    no_exploration_success: float = test_complex_network()  # No Exploration: 0.813
    noisy_inputs_success: float = test_complex_network(fix_inp=False)  # Noisy Inputs: 0.816
    noisy_actions_success: float = test_complex_network(noisy_actions=True)  # Noisy Actions: 0.82
    print(f"No Exploration: {no_exploration_success}")
    print(f"Noisy Inputs: {noisy_inputs_success}")
    print(f"Noisy Actions: {noisy_actions_success}")
