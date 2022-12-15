# py
from math import pi
from typing import List, Tuple, Optional, Any

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, zeros, clip, relu
from torch.optim import Adam, Optimizer
from torch.distributions import Distribution, Uniform

# lib
from base.quantum import Operator
from base.transformer import Transformer
from base.action_space import ActionSpace, RestrictedActionSpace
from base.multi_env import MultiEnv
from base.nash import compute_nash_eq_b
from base.general import static_order_players
from base.utils import calc_dist

# plotting
import matplotlib.pyplot as plt
import tikzplotlib
from numpy import ndarray, linspace, concatenate


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


def train_noisy_inputs(episodes: int,
                       fix_inp: True,
                       fix_inp_time: int,
                       num_players: int,
                       agents: List[Transformer],
                       optimizers: List[Optimizer],
                       env: MultiEnv,
                       reward_distribution: Tensor,
                       player_tokens: Tensor) -> Tensor:
    # compute nash-eq
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


def train_noisy_actions(episodes: int,
                        fix_inp: True,
                        fix_actions_time: int,
                        num_players: int,
                        agents: List[Transformer],
                        optimizers: List[Optimizer],
                        env: MultiEnv,
                        reward_distribution: Tensor,
                        player_tokens: Tensor) -> Tensor:
    # compute nash-eq
    dist_computations: Tensor = zeros((episodes,))
    nash_eq: List[Tuple[Tensor, ...]] = compute_nash_eq_b(env)
    # noise-distribution
    uniform: Distribution = Uniform(-0.25, 0.25)
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

            # add noise
            if step < fix_actions_time:
                noise = uniform.sample((num_players, env.action_space.num_params))
                noise = clip(noise, min=env.action_space.mins, max=env.action_space.maxs)
                for i in range(0, num_players):
                    actions[player_idx] += noise[player_idx]

            # compute q-values
            qs = env.step(*actions)

            # define loss
            loss_idx: Tensor = -qs[player_idx]

            # optimize alice
            optimizers[player_idx].zero_grad()
            loss_idx.backward()
            optimizers[player_idx].step()

    return dist_computations


class Experiment:
    _num_samples: int
    _num_episodes: int
    _name: Optional[str]

    def __init__(self):
        self._num_samples = 15
        self._num_episodes = 15
        self._name = None

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @num_episodes.setter
    def num_episodes(self, value: int):
        self._num_episodes = value

    def _prepare(self) -> Any:
        raise NotImplementedError

    def run(self) -> Tensor:
        raise NotImplementedError


class GradDesc(Experiment):

    def __init__(self):
        super(GradDesc, self).__init__()
        self._num_episodes = 3750
        self._name = "Without Transformer"

    def _prepare(self) -> Tuple[MultiEnv, int]:
        env: MultiEnv = create_dilemma()
        episodes: int = self.num_episodes
        return env, episodes

    @staticmethod
    def _generate_parameters() -> Tensor:
        uniform_theta: Distribution = Uniform(0., pi)
        uniform_phi: Distribution = Uniform(0., pi / 2)
        return tensor([uniform_theta.sample((1,)), uniform_phi.sample((1,))], requires_grad=True)

    def run(self) -> Tensor:
        # get env and episodes
        env, episodes = self._prepare()
        # compute nash eq
        dist_computations: Tensor = zeros((self._num_samples,))
        nash_eq: List[Tuple[Tensor, ...]] = compute_nash_eq_b(env)
        # initialize params and optimizers
        players_params: List[Tensor] = [self._generate_parameters() for _ in range(0, env.num_players)]
        optimizers: List[Optimizer] = [Adam(params=[players_params[i]]) for i in range(0, env.num_players)]
        # loop over episodes
        for step in range(0, episodes - 1):
            # general optimization
            for i in range(0, env.num_players):
                qs: List[Tensor] = env.step(*players_params)
                loss: Tensor = -qs[i]
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
            if step % 250 == 0:
                dist = calc_dist(nash_eq, players_params)
                idx: int = int(step / 250)
                dist_computations[idx] = dist
        return dist_computations


class ZeroExploration(Experiment):

    def __init__(self):
        super(ZeroExploration, self).__init__()
        self._name = "No Exploration Mechanism"

    def _prepare(self) -> Tuple[int, bool, int, List[Transformer], List[Optimizer], MultiEnv, int, Tensor, Tensor]:
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
        fix_inp: bool = True
        fix_inp_time: int = int(episodes * 0.6)

        # inputs
        reward_distribution: Tensor = env.reward_distribution
        player_tokens: Tensor = eye(num_players)
        return (episodes, fix_inp, fix_inp_time, num_players, agents, optimizers,
                env, reward_distribution, player_tokens)

    def run(self) -> Tensor:
        return train_noisy_inputs(*self._prepare())


class NoisyInputs(Experiment):

    def __init__(self):
        super(NoisyInputs, self).__init__()
        self._name = "Noisy Inputs"

    def _prepare(self) -> Tuple[int, bool, int, List[Transformer], List[Optimizer], MultiEnv, int, Tensor, Tensor]:
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
        return (episodes, fix_inp, fix_inp_time, num_players, agents, optimizers,
                env, reward_distribution, player_tokens)

    def run(self) -> Tensor:
        return train_noisy_inputs(*self._prepare())


class NoisyActions(Experiment):

    def __init__(self):
        super(NoisyActions, self).__init__()
        self._name = "Noisy Actions"

    def _prepare(self) -> Tuple[int, bool, int, List[Transformer], List[Optimizer], MultiEnv, int, Tensor, Tensor]:
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
        fix_inp: bool = True
        fix_inp_time: int = int(episodes * 0.6)

        # inputs
        reward_distribution: Tensor = env.reward_distribution
        player_tokens: Tensor = eye(num_players)
        return (episodes, fix_inp, fix_inp_time, num_players, agents, optimizers,
                env, reward_distribution, player_tokens)

    def run(self) -> Tensor:
        return train_noisy_actions(*self._prepare())


def generate_tikz_file(experiments: List[Experiment]):
    # set up plot
    nrows: int = 2
    ncols: int = 2
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_figheight(8.)
    fig.set_figwidth(10.)

    # define experiment hyperparameters
    num_experiments: int = 20
    xs: List[ndarray] = []
    ys: List[ndarray] = []
    stand_devs: List[ndarray] = []
    mins_maxs: List[ndarray] = []

    # conduct experiment
    for experiment in experiments:
        print(experiment.name)
        distances: Tensor = zeros((num_experiments, experiment.num_samples))
        for j in range(0, num_experiments):
            dists: Tensor = experiment.run()
            distances[j] = dists
        # compute metrics
        x: ndarray = linspace(0, experiment.num_episodes - 1, experiment.num_samples)
        y: ndarray = distances.mean(0).numpy()
        stand_dev: ndarray = distances.std(0).numpy()
        min_stand_dev: ndarray = relu(distances.mean(0) - distances).mean(0).view(-1, experiment.num_samples).numpy()
        max_stand_dev: ndarray = relu(distances - distances.mean(0)).mean(0).view(-1, experiment.num_samples).numpy()
        min_max: ndarray = concatenate((min_stand_dev, max_stand_dev))
        # prepare plotting
        xs.append(x)
        ys.append(y)
        stand_devs.append(stand_dev)
        mins_maxs.append(min_max)

    # plot data
    for row in range(0, nrows):
        for col in range(0, ncols):
            idx: int = 2 * row + col
            axs[row, col].errorbar(xs[idx], ys[idx], yerr=mins_maxs[idx],
                                   color='blue', ecolor='black', elinewidth=2, capsize=2., fmt='.k')
            axs[row, col].set_title(experiments[idx].name)

    # title axes
    plt.setp(axs[-1, :], xlabel="Episoden")
    plt.setp(axs[:, 0], ylabel="d(s, s')")

    # show
    # plt.show()

    # save tex file
    tikzplotlib.save("experiments.tex")


if __name__ == '__main__':
    exps: List[Experiment] = [GradDesc(), ZeroExploration(), NoisyActions(), NoisyInputs()]
    generate_tikz_file(exps)


# % add vertical & horizontal zep to plots (60., 35.)
# % change labels to math mode
# % [scale=0.8]