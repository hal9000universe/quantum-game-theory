# py
from typing import List, Tuple
from math import pi

# nn & rl
from torch import Tensor, tensor, zeros, eye, relu, load
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.distributions import Uniform, Distribution

# lib
from base.transformer import Transformer
from base.env import Env
from base.general import static_order_players
from base.utils import calc_dist, create_env, compute_oriented_std, compute_relu_std
from dataset.dataset import GameNashDataset, MicroGameNashDataset
from training.training import get_model_path, get_max_idx
from experiments.prisoners_dilemma import ComplexNetwork

# plotting
from matplotlib.pyplot import subplots, errorbar, xlabel, ylabel, show, setp
from numpy import ndarray, linspace
from tikzplotlib import save


def track_algorithm(model: type = Transformer):
    # define variables
    num_experiments: int = 300
    episodes: int = 15
    plot_frequency: int = 1
    distances: Tensor = zeros((num_experiments, episodes // plot_frequency))

    # training framework
    noisy_inputs: bool = False
    noisy_actions: bool = False

    nash_eq: Tensor = tensor([[0., pi / 2], [0., pi / 2]])
    for game in range(0, num_experiments):
        reward_distribution: Tensor = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])

        # define quantum game
        env: Env = create_env()
        env.reward_distribution = reward_distribution
        num_players: int = env.num_players

        # define and initialize agents
        num_encoder_layers: int = 2
        agents: List[Module] = list()
        optimizers: List[Optimizer] = list()
        for _ in range(0, num_players):
            if model == Transformer:
                player: Transformer = Transformer(
                    num_players=num_players,
                    num_encoder_layers=num_encoder_layers,
                    num_actions=env.action_space.num_params,
                )
            else:
                player: ComplexNetwork = ComplexNetwork()
            # player = load(get_model_path(get_max_idx()))
            optim: Optimizer = Adam(params=player.parameters())
            agents.append(player)
            optimizers.append(optim)

        # define hyperparameters
        fix_inp: bool = not noisy_inputs
        fix_inp_time: int = int(episodes * 0.6)
        fix_actions_time: int = int(episodes * 0.6)

        # inputs
        reward_distribution: Tensor = env.reward_distribution
        player_tokens: Tensor = eye(num_players)

        # initialize noise distribution
        uniform: Distribution = Uniform(-0.2, 0.2)

        # training
        for episode in range(0, episodes):
            # get state from environment
            state = env.reset(fix_inp=fix_inp)

            # sample players for the game
            players, player_indices = static_order_players(num_players, agents)

            for player_idx in player_indices:
                # compute actions
                actions: List[Tensor] = []
                for i, player in enumerate(players):
                    if model == Transformer:
                        params: Tensor = player(state, reward_distribution, player_tokens[i])
                    else:
                        params: Tensor = player(state)
                    if noisy_actions:
                        params += uniform.sample(params.shape)
                    actions.append(params)

                # compute q-values
                qs: List[Tensor] = env.step(*actions)

                # define loss
                loss_idx: Tensor = -qs[player_idx]

                # optimize alice
                optimizers[player_idx].zero_grad()
                loss_idx.backward()
                optimizers[player_idx].step()

            if episode == fix_inp_time:
                fix_inp = True

            if episode == fix_actions_time:
                noisy_actions = False

            if episode % plot_frequency == 0:
                state = env.reset(fix_inp=fix_inp)
                players, player_indices = static_order_players(num_players, agents)
                actions: List[Tuple[Tensor, ...]] = []
                for i, player in enumerate(players):
                    if model == Transformer:
                        params: Tensor = player(state, reward_distribution, player_tokens[i])
                    else:
                        params: Tensor = player(state)
                    actions.append(tuple(params))
                actions: Tensor = tensor(actions)
                distances[game, episode // plot_frequency] = calc_dist(nash_eq, actions, env.action_space.operator)

    # compute plotting data
    x: ndarray = plot_frequency * linspace(0, distances.shape[1] - 1, distances.shape[1])
    y: ndarray = distances.mean(0).numpy()
    # stand_dev: ndarray = compute_oriented_std(distances, 0)
    stand_dev: ndarray = compute_relu_std(distances)

    return x, y, stand_dev


def make_plots():
    # set up plot
    nrows: int = 1
    ncols: int = 2
    fig, axs = subplots(nrows, ncols)
    fig.set_figheight(5.)
    fig.set_figwidth(12.)

    # generate data
    x_cn, y_cn, std_cn = track_algorithm(model=ComplexNetwork)
    x_tr, y_tr, std_tr = track_algorithm(model=Transformer)

    # plot success rate data
    axs[0].errorbar(x_cn, y_cn, yerr=std_cn, color='blue', ecolor='black', elinewidth=2, capsize=2., fmt='.k')
    axs[0].set_title("Distanz zum Nash-Gleichgewicht")

    # plot performance data
    axs[1].errorbar(x_tr, y_tr, yerr=std_tr, color='blue', ecolor='black', elinewidth=2, capsize=2., fmt='.k')
    axs[1].set_title("Distanz zum Nash-Gleichgewicht")

    # label axes
    setp(axs, xlabel="Episoden")
    setp(axs[0], ylabel="d(s, s')")
    setp(axs[1], ylabel="d(s, s')")

    # show()

    save("experiments/plots/architectures.tex")


if __name__ == '__main__':
    make_plots()
