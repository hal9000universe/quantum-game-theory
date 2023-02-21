# py
from math import pi
from typing import Tuple, List, Callable
from copy import copy

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, load, zeros
from torch.optim import Adam, Optimizer, AdamW
from torch.distributions import Distribution, Uniform

# lib
from base.quantum import Operator
from base.env import Env
from base.action_space import ActionSpace, RestrictedActionSpace
from base.transformer import Transformer
from training.training import get_max_idx, get_mixed_model_path


"""This file is all about the maddpg algorithm."""


def static_order_players(num_players: int, agents: List[Transformer]) -> Tuple[List[Transformer], List[int]]:
    return agents[0:num_players], [i for i in range(0, num_players)]


def train(episodes: int,
          fix_inp: bool,
          fix_inp_time: int,
          noisy_actions: bool,
          fix_actions_time: int,
          num_players: int,
          agents: List[Transformer],
          optimizers: List[Optimizer],
          env: Env,
          reward_distribution: Tensor,
          player_tokens: Tensor) -> List[Transformer]:
    """The train function is an implementation of the
    Multi-Agent Deep Deterministic Policy Gradient Algorithm for quantum games.

    Keyword arguments
    episodes: int specifying how many games are to be played.
    fix_inp: bool specifying whether to add noise to the inputs as an exploration mechanism.
    fix_inp_time: int specifying when to remove noise from the inputs (in case noise is used).
    noisy_actions: bool specifying whether to add noise the actions as an exploration mechanism.
    fix_actions_time: int specifying when to remove noise from the actions (in case noise is used).
    num_players: int specifying the number of players needed for the given quantum game.
    agents: a list of transformers which will play the game.
    optimizers: a list of optimizers which will be applied to the agents.
    env: Env handling the logic of the quantum game and the q-value computations.
    reward_distribution: Tensor specifying the reward distribution of the quantum game to be played.
    player_tokens: Tensor containing one_hot vectors indicating player positions."""
    # initialize noise distribution
    uniform: Distribution = Uniform(-0.25, 0.25)
    # initialize variables for an early stopping mechanism
    last_action: Tensor = zeros((env.action_space.num_params,))
    new_action: Tensor
    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # sample players for the game
        players, player_indices = static_order_players(num_players, agents)

        copy_actions: List[Tensor] = list()
        for player_idx in player_indices:
            actions: List[Tensor] = []
            # compute actions
            for i, player in enumerate(players):
                params: Tensor = player(state, reward_distribution, player_tokens[i])
                if noisy_actions:
                    params += uniform.sample(params.shape)
                actions.append(params)

            # compute q-values
            qs = env.step(*actions)

            # define loss
            loss_idx: Tensor = -qs[player_idx]

            # optimize alice
            optimizers[player_idx].zero_grad()
            loss_idx.backward()
            optimizers[player_idx].step()

            copy_actions = copy(actions)

        # compute significance of action change
        new_action = copy_actions[0]
        dist = (new_action - last_action).norm()
        if dist < 1e-7:  # early stopping
            print("Stopped training early.")
            break

        if step == fix_inp_time:
            fix_inp = True

        if step == fix_actions_time:
            noisy_actions = False

    return agents


def training_framework(reward_distribution: Tensor,
                       load_model: bool = False,
                       noisy_inputs: bool = True,
                       noisy_actions: bool = False,
                       model_path_fun: Callable[[int], str] = get_mixed_model_path) -> Tensor:
    """The training_framework function serves to eliminate or shorten repeated code (DRY)"""
    # define quantum game
    num_players: int = 2
    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    mat: Tensor = matrix_exp(-1j * gamma * kron(D, D) / 2)
    J: Operator = Operator(mat=mat)
    action_space: ActionSpace = RestrictedActionSpace()

    # initialize env
    env: Env = Env(num_players=num_players)
    env.J = J
    env.reward_distribution = reward_distribution
    env.action_space = action_space

    # define and initialize agents
    num_encoder_layers: int = 2
    agents: List[Transformer] = list()
    optimizers: List[Optimizer] = list()
    for _ in range(0, num_players):
        if load_model:
            player: Transformer = load(model_path_fun(get_max_idx(model_path_fun)))
        else:
            player: Transformer = Transformer(
                num_players=num_players,
                num_encoder_layers=num_encoder_layers,
                num_actions=env.action_space.num_params,
            )
        optim: Optimizer = AdamW(params=player.parameters())
        agents.append(player)
        optimizers.append(optim)

    # define hyperparameters
    episodes: int = 50
    fix_inp: bool = not noisy_inputs
    fix_inp_time: int = int(episodes * 0.6)
    fix_actions_time: int = int(episodes * 0.6)

    # inputs
    reward_distribution: Tensor = env.reward_distribution
    player_tokens: Tensor = eye(num_players)

    # training
    agents: List[Transformer] = train(
        episodes=episodes,
        fix_inp=fix_inp,
        fix_inp_time=fix_inp_time,
        noisy_actions=noisy_actions,
        fix_actions_time=fix_actions_time,
        agents=agents,
        optimizers=optimizers,
        env=env,
        num_players=num_players,
        reward_distribution=reward_distribution,
        player_tokens=player_tokens,
    )

    # return final actions
    state = env.reset(fix_inp=fix_inp)
    players, player_indices = static_order_players(num_players, agents)
    actions: List[Tuple[Tensor, ...]] = []
    for i, player in enumerate(players):
        params: Tensor = player(state, reward_distribution, player_tokens[i])
        actions.append(tuple(params))
    actions: Tensor = tensor(actions)
    return actions
