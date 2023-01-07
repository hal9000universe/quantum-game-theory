# py
from typing import Callable

from math import pi

# nn & rl
from torch import Tensor, tensor, load, eye
from torch.optim import Adam

# lib
from base.utils import calc_dist, create_env
from base.general import training_framework, train, static_order_players
from base.action_space import RestrictedActionSpace
from base.env import Env
from dataset.dataset import MicroGameNashDataset
from experiments.complex_network import ComplexNetwork


def test_complex_network() -> float:
    ds: MicroGameNashDataset = load("dataset/game-nash-datasets/game-nash-dataset-125.pth")
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator

    # define counting variables
    num_games: int = len(ds)
    num_successes: int = 0

    for i, (reward_distribution, nash_eq) in enumerate(ds, start=1):
        # initialize environment variables
        env: Env = create_env()
        player_tokens: Tensor = eye(2)

        # initialize agents and optimizers
        agents: List[ComplexNetwork] = list()
        optimizers: List[Optimizer] = list()
        for player in range(0, env.num_players):
            agent: ComplexNetwork = ComplexNetwork()
            optimizer: Optimizer = Adam(params=agent.parameters())
            agents.append(agent)
            optimizers.append(optimizer)

        # training
        agents = train(
            episodes=100,
            fix_inp=True,
            fix_inp_time=0,
            noisy_actions=False,
            fix_actions_time=0,
            num_players=env.num_players,
            agents=agents,
            optimizers=optimizers,
            env=env,
            reward_distribution=reward_distribution,
            player_tokens=player_tokens,
        )

        # compute actions
        state = env.reset(True)
        players, player_indices = static_order_players(env.num_players, agents)
        actions: List[Tuple[Tensor, ...]] = []
        for player_idx, player in enumerate(players):
            params: Tensor = player(state, reward_distribution, player_tokens[player_idx])
            actions.append(tuple(params))
        actions: Tensor = tensor(actions)
        dist: Tensor = calc_dist(nash_eq, actions, parametrization)
        if dist < 0.2:
            num_successes += 1
        print(f"Game: {i} - Dist: {dist} - Successes: {num_successes}")

    return num_successes / num_games


def test_transformer(load_model: bool = False, noisy_inputs: bool = False, noisy_actions: bool = False) -> float:
    ds: MicroGameNashDataset = load("dataset/game-nash-datasets/game-nash-dataset-125.pth")
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator

    num_games: int = len(ds)
    num_successes: int = 0

    for i, (reward_distribution, nash_eq) in enumerate(ds, start=1):
        actions: Tensor = training_framework(
            reward_distribution=reward_distribution,
            load_model=load_model,
            noisy_inputs=noisy_inputs,
            noisy_actions=noisy_actions,
        )
        dist: Tensor = calc_dist(nash_eq, actions, parametrization)
        if dist < 0.2:
            num_successes += 1
        print(f"Game: {i} - Dist: {dist} - Successes: {num_successes}")

    return num_successes / num_games


def complex_network_test():
    # Success rate: 0.05
    accuracy: float = test_complex_network()
    print(f"Complex Network Accuracy: {accuracy}")


def no_exploration_test():
    # Success rate: 0.43 - 0.46
    accuracy: float = test_transformer(
        load_model=False,
        noisy_inputs=False,
        noisy_actions=False,
    )
    print(f"No Exploration Accuracy: {accuracy}")


def pretrained_no_exploration_test():
    # Success rate: 0.82 - 0.87
    accuracy: float = test_transformer(
        load_model=True,
        noisy_inputs=False,
        noisy_actions=False,
    )
    print(f"Pretrained No Exploration Accuracy: {accuracy}")


def noisy_inputs_test():
    # Success rate: 0.43 - 0.45
    accuracy: float = test_transformer(
        load_model=False,
        noisy_inputs=True,
        noisy_actions=False,
    )
    print(f"Noisy Inputs Accuracy: {accuracy}")


def pretrained_noisy_inputs_test():
    # Success rate: 0.85 - 0.88
    accuracy: float = test_transformer(
        load_model=True,
        noisy_inputs=True,
        noisy_actions=False,
    )
    print(f"Pretrained Noisy Inputs Accuracy: {accuracy}")


def noisy_actions_test():
    # Success rate: 0.41 - 0.47
    accuracy: float = test_transformer(
        load_model=False,
        noisy_inputs=False,
        noisy_actions=True,
    )
    print(f"Noisy Actions Accuracy: {accuracy}")


def pretrained_noisy_actions_test():
    # Success rate: 0.88 - 0.89
    accuracy: float = test_transformer(
        load_model=True,
        noisy_inputs=False,
        noisy_actions=True,
    )
    print(f"Pretrained Noisy Actions Accuracy: {accuracy}")


if __name__ == '__main__':
    complex_network_test()
    no_exploration_test()
    pretrained_no_exploration_test()
    noisy_inputs_test()
    pretrained_noisy_inputs_test()
    noisy_actions_test()
    pretrained_noisy_actions_test()
