# py
from math import pi
from typing import List, Tuple, Optional, Any

# nn & rl
from torch import tensor, Tensor, kron, complex64, eye, matrix_exp, zeros
from torch.optim import Adam, Optimizer
from torch.distributions import Distribution, Uniform

# lib
from quantum import Operator
from transformer import Transformer
from env import ActionSpace, GeneralActionSpace, RestrictedActionSpace
from multi_env import MultiEnv
from nash import compute_nash_eq_b
from general import static_order_players

# plotting
import matplotlib.pyplot as plt
import tikzplotlib
from numpy import ndarray, linspace


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
        return train(*self._prepare())


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


# TODO: fix error-range (metric should not go below 0.)
class GradDesc(Experiment):

    def __init__(self):
        super(GradDesc, self).__init__()
        self._num_episodes = 3750
        self._name = "Gradient Descent"

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


# TODO: implement UniformNoisyActions Experiment
# TODO: Ornstein-Uhlenbeck Process NoisyActions Experiment


def generate_tikz_code() -> str:
    # define experiment
    num_experiments: int = 2
    # conduct experiment
    experiment: Experiment = NoisyInputs()
    distances: Tensor = zeros((num_experiments, experiment.num_episodes))
    for j in range(0, num_experiments):
        dists: Tensor = experiment.run()
        distances[j] = dists

    # convert to tikz format
    average: Tensor = distances.mean(0)
    stand_dev: Tensor = distances.std(0)
    tikz: str = "x xerr y yerr class \n"
    for j in range(0, experiment.num_episodes):
        tikz = tikz + f"{i}" + f" {0}" + f" {average[j]}" + f" {stand_dev[j]}" + f" {6}" + " \n"
    print(tikz)
    return tikz


def generate_plot(experiment: Experiment, distances: Tensor):
    # convert to tikz format
    x: ndarray = linspace(0, experiment.num_episodes - 1, experiment.num_episodes)
    y: ndarray = distances.mean(0).numpy()
    stand_dev: ndarray = distances.std(0).numpy()
    plt.errorbar(x, y, yerr=stand_dev, color='blue', ecolor='black', elinewidth=2, capsize=2., fmt='.k')
    plt.xlabel("Episoden")
    plt.ylabel("Distanz zum Nash-Gleichgewicht")
    plt.show()


def generate_tikz_file(experiments: List[Experiment]):
    # set up plot
    nrows: int = 2
    ncols: int = 2
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_figheight(8.)
    fig.set_figwidth(10.)

    # define experiment hyperparameters
    num_experiments: int = 10
    xs: List[ndarray] = []
    ys: List[ndarray] = []
    stand_devs: List[ndarray] = []

    # conduct experiment
    for experiment in experiments:
        distances: Tensor = zeros((num_experiments, experiment.num_samples))
        for j in range(0, num_experiments):
            dists: Tensor = experiment.run()
            distances[j] = dists
        x: ndarray = linspace(0, experiment.num_episodes - 1, experiment.num_samples)
        y: ndarray = distances.mean(0).numpy()
        stand_dev: ndarray = distances.std(0).numpy()
        xs.append(x)
        ys.append(y)
        stand_devs.append(stand_dev)

    # plot data
    for row in range(0, nrows):
        for col in range(0, ncols):
            axs[row, col].errorbar(xs[row + col], ys[row + col], yerr=stand_devs[row + col],
                                   color='blue', ecolor='black', elinewidth=2, capsize=2., fmt='.k')
            axs[row, col].set_title(experiments[row + col].name)

    # title axes
    plt.setp(axs[-1, :], xlabel="Episoden")
    plt.setp(axs[:, 0], ylabel="d(s, s')")

    # show
    plt.show()

    # save tex file
    tikzplotlib.save("experiments.tex")


if __name__ == '__main__':
    exps: List[Experiment] = [NoisyInputs(), GradDesc(), NoisyInputs(), GradDesc()]
    generate_tikz_file(exps)
