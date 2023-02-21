# py
from typing import Callable

# nn & rl
from torch import Tensor, tensor, eye
from torch.optim import Adam

# lib
from base.nash import is_nash
from base.utils import calc_dist, create_env
from base.general import training_framework, train, static_order_players
from base.action_space import RestrictedActionSpace
from base.env import Env
from dataset.dataset import MicroGameNashDataset, GameNashDataset, get_mixed_gn_path
from training.training import path_fun_map
from experiments.prisoners_dilemma import ComplexNetwork


"""This file compares the ComplexNetwork and Transformer architectures on a variety of quantum games."""


def test_complex_network() -> float:
    """The test_complex_network function applies the maddpg algorithm to a ComplexNetwork on a variety
    of quantum games and computes the success frequency."""
    ds: MicroGameNashDataset = GameNashDataset(0.9, 1.0)
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


def test_transformer(load_model: bool = False, noisy_inputs: bool = False, noisy_actions: bool = False,
                     path_fun: Callable[[int], str] = get_mixed_gn_path) -> float:
    """The test_transformer function applies the maddpg algorithm to a transformer
    on a variety of quantum games and computes the success frequency."""
    ds: MicroGameNashDataset = GameNashDataset(0.9, 1.0, path_fun=path_fun)
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator

    num_games: int = len(ds)
    num_successes: int = 0
    num_nash_successes: int = 0

    for i, (reward_distribution, nash_eq) in enumerate(ds, start=1):
        env: Env = create_env(reward_distribution)
        env.action_space.num_steps = 30
        actions: Tensor = training_framework(
            reward_distribution=reward_distribution,
            load_model=load_model,
            noisy_inputs=noisy_inputs,
            noisy_actions=noisy_actions,
            model_path_fun=path_fun_map(path_fun),
        )
        dist: Tensor = calc_dist(nash_eq, actions, parametrization)
        if dist < 0.2:
            num_successes += 1
        if is_nash(list(tuple(actions)), env):
            num_nash_successes += 1
        print(f"Game: {i} - Dist: {dist} - Successes: {num_successes} - Nash Successes: {num_nash_successes}")

    return num_successes / num_games


if __name__ == '__main__':
    complex_network_success: float = test_complex_network()
    transformer_success: float = test_transformer(load_model=True, path_fun=get_mixed_gn_path)
    print(f"ComplexNetwork: {complex_network_success}")
    print(f"Transformer: {transformer_success}")
