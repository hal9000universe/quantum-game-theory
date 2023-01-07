# py
from math import pi
from typing import Tuple, List

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, load
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.distributions import Distribution, Uniform

# lib
from base.quantum import Operator
from base.env import Env
from base.action_space import ActionSpace, RestrictedActionSpace
from base.transformer import Transformer
from base.utils import create_env
from training.training import get_max_idx, get_model_path


def static_order_players(num_players: int, agents: List[Module]) -> Tuple[List[Module], List[int]]:
    return agents[0:num_players], [i for i in range(0, num_players)]


def train(episodes: int,
          fix_inp: bool,
          fix_inp_time: int,
          noisy_actions: bool,
          fix_actions_time: int,
          num_players: int,
          agents: List[Module],
          optimizers: List[Optimizer],
          env: Env,
          reward_distribution: Tensor,
          player_tokens: Tensor,
          nisq: bool = False) -> List[Module]:
    # initialize noise distribution
    uniform: Distribution = Uniform(-0.2, 0.2)
    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # sample players for the game
        players, player_indices = static_order_players(num_players, agents)

        for player_idx in player_indices:
            # compute actions
            actions: List[Tensor] = []
            for i, player in enumerate(players):
                params: Tensor = player(state, reward_distribution, player_tokens[i])
                if noisy_actions:
                    params += uniform.sample(params.shape)
                actions.append(params)

            # compute q-values
            if nisq:
                qs: List[Tensor] = env.q_step(*actions)
            else:
                qs: List[Tensor] = env.step(*actions)

            # define loss
            loss_idx: Tensor = -qs[player_idx]

            # optimize alice
            optimizers[player_idx].zero_grad()
            loss_idx.backward()
            optimizers[player_idx].step()

        if step == fix_inp_time:
            fix_inp = True

        if step == fix_actions_time:
            noisy_actions = False

    return agents


def training_framework(reward_distribution: Tensor,
                       load_model: bool = False,
                       noisy_inputs: bool = True,
                       noisy_actions: bool = False) -> Tensor:
    # define quantum game
    env: Env = create_env()
    env.reward_distribution = reward_distribution
    num_players: int = env.num_players

    # define and initialize agents
    num_encoder_layers: int = 2
    agents: List[Transformer] = list()
    optimizers: List[Optimizer] = list()
    for _ in range(0, num_players):
        if load_model:
            player: Transformer = load(get_model_path(get_max_idx()))
        else:
            player: Transformer = Transformer(
                num_players=num_players,
                num_encoder_layers=num_encoder_layers,
                num_actions=env.action_space.num_params,
            )
        optim: Optimizer = Adam(params=player.parameters())
        agents.append(player)
        optimizers.append(optim)

    # define hyperparameters
    episodes: int = 100
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
