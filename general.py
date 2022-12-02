# py
from math import pi
from typing import Optional, Tuple, Callable, List
from random import randint

# nn & rl
from torch import tensor, Tensor, cat, kron, real, ones, complex64, float32, allclose, eye
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Optimizer
from torch.distributions import Uniform, Distribution

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import QuantumSystem, Operator
from multi_env import MultiEnv
from transformer import Transformer
from env import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from qmain import Env


def sample_players(num_players: int, agents: List[Transformer]) -> Tuple[List[Transformer], List[int]]:
    players: List[Transformer] = list()
    player_indices: List[int] = list()
    for _ in range(0, num_players):
        rd_idx: int = randint(0, len(agents) - 1)
        players.append(agents[rd_idx])
        player_indices.append(rd_idx)
    return players, player_indices


def static_order_players(num_players: int, agents: List[Transformer]) -> Tuple[List[Transformer], List[int]]:
    return agents[0:num_players], [i for i in range(0, num_players)]


def train(episodes: int,
          fix_inp: True,
          fix_inp_time: int,
          num_players: int,
          agents: List[Transformer],
          optimizers: List[Optimizer],
          env: MultiEnv,
          reward_distribution: Tensor,
          player_tokens: Tensor) -> List[Transformer]:
    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # sample players for the game
        # players, player_indices = sample_players(num_players, agents)
        players, player_indices = static_order_players(num_players, agents)

        for player_idx in player_indices:
            # compute actions
            actions: List[Tensor] = []
            for i, player in enumerate(players):
                params: Tensor = player(state, reward_distribution, player_tokens[i])
                actions.append(params)

            # compute q-values
            qs = env.step(*actions)

            # define loss
            loss_idx: Tensor = -qs[player_idx]

            # optimize alice
            optimizers[player_idx].zero_grad()
            loss_idx.backward()
            optimizers[player_idx].step()

        if step == fix_inp_time:
            fix_inp = True

    return agents


def main() -> List[Tensor]:
    # define quantum game
    num_players: int = 2
    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    mat: Tensor = matrix_exp(-1j * gamma * kron(D, D) / 2)
    J: Operator = Operator(mat=mat)
    reward_distribution: Tensor = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])
    action_space: ActionSpace = RestrictedActionSpace()

    # initialize env
    env: Env = MultiEnv(num_players=num_players)
    env.J = J
    env.reward_distribution = reward_distribution
    env.action_space = action_space

    # define and initialize agents
    num_encoder_layers: int = 2
    agents: List[Transformer] = list()
    optimizers: List[Optimizer] = list()
    for _ in range(0, num_players):
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
    fix_inp: bool = False
    fix_inp_time: int = int(episodes * 0.6)

    # inputs
    reward_distribution: Tensor = env.reward_distribution
    player_tokens: Tensor = eye(num_players)

    # training
    agents: List[Transformer] = train(
        episodes=episodes,
        fix_inp=fix_inp,
        fix_inp_time=fix_inp_time,
        agents=agents,
        optimizers=optimizers,
        env=env,
        num_players=num_players,
        reward_distribution=reward_distribution,
        player_tokens=player_tokens,
    )

    # monitoring
    print('-----')
    state = env.reset(fix_inp=fix_inp)
    players, player_indices = static_order_players(num_players, agents)
    actions: List[Tensor] = []
    for i, player in enumerate(players):
        params: Tensor = player(state, reward_distribution, player_tokens[i])
        actions.append(params)
    qs = env.step(*actions)
    print(f"Actions: {actions}")
    print(f"Rewards: {qs}")
    return actions


if __name__ == '__main__':
    main()

# result (transformer, static_order): 1. success rate
# result (transformer, only self-play): 0.55 success rate
