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
from matplotlib.pyplot import subplots, errorbar, show, setp
from numpy import array, ndarray, linspace
from tikzplotlib import save


FAILS: bool = False


def track_algorithm(pretrained: bool = False):
    # define variables
    test_ds: GameNashDataset = GameNashDataset(0.999, 1.)
    print(f"Number of Quantum Games: {len(test_ds)}")
    num_experiments: int = len(test_ds)
    episodes: int = 15
    plot_frequency: int = 1
    distances: Tensor = zeros((num_experiments, episodes // plot_frequency))
    to_print: List[bool] = list()

    # training framework
    noisy_inputs: bool = False
    noisy_actions: bool = False

    for game, (reward_distribution, nash_eq) in enumerate(test_ds):
        print(f"Game: {game}")
        # define quantum game
        env: Env = create_env()
        env.reward_distribution = reward_distribution
        num_players: int = env.num_players

        # define and initialize agents
        num_encoder_layers: int = 2
        agents: List[Module] = list()
        optimizers: List[Optimizer] = list()
        for _ in range(0, num_players):
            if pretrained:
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
                    params: Tensor = player(state, reward_distribution, player_tokens[i])
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
                    params: Tensor = player(state, reward_distribution, player_tokens[i])
                    actions.append(tuple(params))
                actions: Tensor = tensor(actions)
                distances[game, episode // plot_frequency] = calc_dist(nash_eq, actions, env.action_space.operator)

            if episode == episodes - 1:
                if distances[game, episode // plot_frequency] < 0.2:
                    to_print.append(True)
                else:
                    to_print.append(FAILS)  # set to True/False to change the behaviour of the graph

    # compute plotting data
    x: ndarray = plot_frequency * linspace(0, distances.shape[1] - 1, distances.shape[1])
    ys: List[Tuple[Tensor, ...]] = list()
    for idx, game_success_bool in enumerate(to_print):
        if game_success_bool:
            ys.append(tuple(distances[idx]))
    ys_arr: Tensor = tensor(ys)
    y: ndarray = ys_arr.mean(0).numpy()
    stand_dev: ndarray = compute_relu_std(ys_arr)

    return x, y, stand_dev


def make_plots():
    # set up plot
    nrows: int = 1
    ncols: int = 2
    fig, axs = subplots(nrows, ncols)
    fig.set_figheight(5.)
    fig.set_figwidth(12.)

    # generate data
    x_scratch, y_scratch, std_scratch = track_algorithm(pretrained=False)
    x_pretrained, y_pretrained, std_pretrained = track_algorithm(pretrained=True)

    # plot success rate data
    axs[0].errorbar(x_scratch, y_scratch, yerr=std_scratch, color='blue', ecolor='black',
                    elinewidth=2, capsize=2., fmt='.k')
    axs[0].set_title("Ohne Pre-Training")

    # plot performance data
    axs[1].errorbar(x_pretrained, y_pretrained, yerr=std_pretrained, color='blue',
                    ecolor='black', elinewidth=2, capsize=2., fmt='.k')
    axs[1].set_title("Mit Pre-Training")

    if FAILS:
        # adjust axis labels
        locs = axs[0].get_yticks()
        labels = axs[0].get_yticklabels()
        axs[1].set_yticks(locs)
        axs[1].set_yticklabels(labels)

    # label axes
    setp(axs, xlabel="Episoden")
    setp(axs[0], ylabel="d(s, s')")
    setp(axs[1], ylabel="d(s, s')")

    # show()

    save("experiments/plots/maddpg_performance.tex")


if __name__ == '__main__':
    make_plots()
