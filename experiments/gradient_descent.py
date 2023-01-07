# py
from math import pi
from typing import Optional, List, Callable

# nn & rl
from torch import tensor, Tensor, row_stack, load
from torch.optim import Adam, Optimizer
from torch.distributions import Uniform, Distribution

# lib
from base.utils import create_env, calc_dist
from base.env import Env
from base.action_space import RestrictedActionSpace
from dataset.dataset import MicroGameNashDataset


def generate_parameters() -> Tensor:
    uniform_theta: Distribution = Uniform(0., pi)
    uniform_phi: Distribution = Uniform(0., pi/2)
    return tensor([uniform_theta.sample((1,)), uniform_phi.sample((1,))], requires_grad=True)


def perform_grad_desc(env: Env,
                      episodes: int,
                      players_params: Optional[List[Tensor]] = None,
                      optimizers: Optional[List[Optimizer]] = None) -> Tensor:
    # initialize params and optimizers
    if players_params is None:
        assert optimizers is None
        players_params: List[Tensor] = [generate_parameters() for _ in range(0, env.num_players)]
        optimizers: List[Optimizer] = [Adam(params=[players_params[i]]) for i in range(0, env.num_players)]
    # loop over episodes
    for step in range(1, episodes):
        # general optimization
        for i in range(0, env.num_players):
            qs: List[Tensor] = env.step(*players_params)
            loss: Tensor = -qs[i]
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
    return row_stack(tuple(players_params))


# 0.46
def test_prisoners_grad_desc() -> float:
    num_successes: int = 0
    env: Env = create_env()
    nash_eq: Tensor = tensor([[0., pi / 2], [0., pi / 2]])

    for attempt in range(0, 100):
        actions: Tensor = perform_grad_desc(env, 4000)
        dist: float = calc_dist(nash_eq, actions, env.action_space.operator)
        if dist < 0.2:
            num_successes += 1
        print(f"Attempt: {attempt} - Dist: {dist} - Successes: {num_successes}")

    return num_successes / 100


# 0.15
def test_grad_desc() -> float:
    ds = load("dataset/game-nash-datasets/game-nash-dataset-125.pth")
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator

    num_games: int = len(ds)
    num_successes: int = 0

    env: Env = create_env()

    for i, (reward_distribution, nash_eq) in enumerate(ds, start=1):
        env.reward_distribution = reward_distribution
        actions: Tensor = perform_grad_desc(env, 4000)
        dist: Tensor = calc_dist(nash_eq, actions, parametrization)
        if dist < 0.2:
            num_successes += 1
        print(f"Game: {i} - Dist: {dist} - Successes: {num_successes}")

    return num_successes / num_games


if __name__ == '__main__':
    prisoners_accuracy: float = test_prisoners_grad_desc()
    general_accuracy: float = test_grad_desc()
    print(f"Prisoner's Accuracy: {prisoners_accuracy}")
    print(f"General Accuracy: {general_accuracy}")
