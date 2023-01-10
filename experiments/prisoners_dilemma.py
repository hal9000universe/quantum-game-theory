# py
from typing import List
from math import pi

# nn & rl
from torch import Tensor, tensor, eye
from torch.optim import Optimizer, Adam

# lib
from base.env import Env
from base.general import train
from base.utils import create_env, calc_dist
from base.complex_network import ComplexNetwork
from base.transformer import Transformer


def test_complex_network(fix_inp: bool = True, noisy_actions: bool = False) -> float:
    num_successes: int = 0
    num_attempts: int = 1000
    for attempt in range(0, num_attempts):
        env: Env = create_env()
        env.reward_distribution = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])
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


def test_transformer(fix_inp: bool = True, noisy_actions: bool = False) -> float:
    # variables
    num_successes: int = 0
    num_attempts: int = 1000

    # hyperparameters
    num_encoder_layers: int = 2
    for attempt in range(0, num_attempts):
        env: Env = create_env()
        num_players: int = 2
        agents: List[Transformer] = [Transformer(
            num_players=env.num_players,
            num_encoder_layers=num_encoder_layers,
            num_actions=env.action_space.num_params,
        ) for _ in range(0, num_players)]
        optimizers: List[Optimizer] = [Adam(params=agents[i].parameters()) for i in range(0, num_players)]
        episodes: int = 3000
        agents: List[Transformer] = train(
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
    complex_network_success: float = test_complex_network()
    # transformer_success: float = test_transformer()
    print(f"ComplexNetwork: {complex_network_success}")
    # print(f"Transformer: {transformer_success}")
