# py
from math import pi
from typing import List, Tuple

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, zeros
from torch.optim import Adam, Optimizer

# lib
from quantum import Operator
from transformer import Transformer
from env import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from multi_env import MultiEnv
from nash import compute_nash_eq_b
from general import static_order_players


def calc_dist(nash_eq: List[Tuple[Tensor, ...]], params: List[Tensor]) -> Tensor:
    nash_eq_t: List = []
    params_t: List = []
    for elem in nash_eq:
        for item in elem:
            nash_eq_t.append(item)
    for elem in params:
        for item in elem:
            params_t.append(item)
    nash_eq_tensor: Tensor = tensor(nash_eq_t)
    params_tensor: Tensor = tensor(params_t)
    diff: Tensor = nash_eq_tensor - params_tensor
    dist: Tensor = diff.square().sum().sqrt()
    return dist


def create_dilemma() -> MultiEnv:
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
    return env


def train(episodes: int,
          fix_inp: True,
          fix_inp_time: int,
          num_players: int,
          agents: List[Transformer],
          optimizers: List[Optimizer],
          env: MultiEnv,
          reward_distribution: Tensor,
          player_tokens: Tensor) -> Tensor:
    dist_computations: Tensor = zeros((episodes,))
    nash_eq: List[Tuple[Tensor, ...]] = compute_nash_eq_b(env)
    # loop over episodes
    for step in range(0, episodes):
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

            # compute distance
            dist = calc_dist(nash_eq, actions)
            dist_computations[step] = dist

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

    return dist_computations


class Experiment:
    _num_episodes: int

    def __init__(self):
        self._num_episodes = 25

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @num_episodes.setter
    def num_episodes(self, value: int):
        self._num_episodes = value

    def prepare(self) -> Tuple[int, bool, int, List[Transformer], List[Optimizer], MultiEnv, int, Tensor, Tensor]:
        raise NotImplementedError

    def run(self) -> Tensor:
        return train(*self.prepare())


class QuantumTransformerNoisyInputsExp(Experiment):

    def prepare(self) -> Tuple[int, bool, int, List[Transformer], List[Optimizer], MultiEnv, int, Tensor, Tensor]:
        # define env
        num_players: int = 2
        env = create_dilemma()

        # define agents and optimizers
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
        episodes: int = self._num_episodes
        fix_inp: bool = False
        fix_inp_time: int = int(episodes * 0.6)

        # inputs
        reward_distribution: Tensor = env.reward_distribution
        player_tokens: Tensor = eye(num_players)
        return [episodes, fix_inp, fix_inp_time, num_players, agents, optimizers,
                env, reward_distribution, player_tokens]


if __name__ == '__main__':
    # find nash equilibrium
    environment: MultiEnv = create_dilemma()
    nash_equilibrium: List[Tuple[Tensor, ...]] = compute_nash_eq_b(environment)
    # define experiment
    num_experiments: int = 2
    # conduct experiment
    experiment: Experiment = QuantumTransformerNoisyInputsExp()
    distances: Tensor = zeros((num_experiments, experiment.num_episodes))
    for i in range(0, num_experiments):
        dists: Tensor = experiment.run()
        distances[i] = dists

    # convert to tikz format
    average: Tensor = distances.mean(0)
    stand_dev: Tensor = distances.std(0)
    tikz: str = "x xerr y yerr class \n"
    for i in range(0, experiment.num_episodes):
        tikz = tikz + f"{i}" + f" {0}" + f" {average[i]}" + f" {stand_dev[i]}" + f" {6}" + " \n"
    print(tikz)

