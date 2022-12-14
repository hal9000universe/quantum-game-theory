# py
from math import pi
from typing import Tuple, List

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, allclose
from torch.optim import Adam, Optimizer

# lib
from quantum import Operator
from multi_env import MultiEnv
from action_space import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from transformer import Transformer


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
    # reward_distribution: Tensor = tensor([[6., 6.], [2., 8.], [8., 2.], [0., 0.]])
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
    episodes: int = 500
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


def check(final_params: Tensor) -> bool:
    learned: bool = allclose(final_params, tensor([0., pi / 2]),
                             rtol=0.05,
                             atol=0.05)
    return learned


def sim_success_evaluation():
    nums: int = 0
    times: int = 100
    for time in range(times):
        final_params1, final_params2 = main()
        if check(final_params1) and check(final_params2):
            print('training successful ...')
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


if __name__ == '__main__':
    sim_success_evaluation()

# result (transformer, static_order): 1. success rate
# result (transformer, only self-play): 0.55 success rate
# result (transformer, random_order): 0.45 success rate
