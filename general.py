# py
from typing import List
from random import randint
from warnings import simplefilter

# nn & rl
from torch import Tensor, tensor, eye
from torch.optim import Optimizer, Adam

# lib
from transformer import Transformer
from multi_env import MultiEnv
from env import ActionSpace, GeneralActionSpace, RestrictedActionSpace


def main():
    simplefilter('ignore', UserWarning)

    # Define environment variables
    num_players: int = 2
    action_space: ActionSpace = RestrictedActionSpace()
    reward_distribution: Tensor = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])
    player_tokens: Tensor = eye(num_players)

    # Configure environment
    env: MultiEnv = MultiEnv(num_players=num_players)
    env.action_space = action_space
    env.reward_distribution = reward_distribution

    # Define agents
    num_encoder_layers: int = 2
    agents: List[Transformer] = []
    agent_indices: List[int] = [i for i in range(0, num_players)]
    for _ in range(0, num_players):
        agent: Transformer = Transformer(
            num_players=num_players,
            num_encoder_layers=num_encoder_layers,
            num_actions=action_space.num_params,
        )
        agents.append(agent)

    # Define learning hyperparameters
    num_episodes: int = 1000
    fix_inp: bool = False
    fix_inp_time: int = int(num_episodes * 0.6)

    # set up optimizer
    optimizers: List[Optimizer] = []
    for i in range(0, num_players):
        optimizer: Optimizer = Adam(params=agents[i].parameters())
        optimizers.append(optimizer)

    # Training loop
    for episode in range(0, num_episodes):
        # generate agent order (SHUFFLE)
        ordered_agents: List[Transformer] = []
        order: List[int] = []
        for i in range(0, num_players):
            rd_idx: int = randint(0, num_players - i - 1)
            agent: Transformer = agents[agent_indices[rd_idx]]
            ordered_agents.append(agent)
            order.append(agent_indices[rd_idx])
            agent_indices.pop(rd_idx)

        # optimization
        for i in range(0, num_players):
            # compute actions
            state: Tensor = env.reset(fix_inp=fix_inp)
            actions: List[Tensor] = []
            for player in range(0, num_players):
                action_params: Tensor = ordered_agents[player](state, reward_distribution, player_tokens[player])
                actions.append(action_params)
            # get optimizer corresponding to agent
            agent_idx: int = order[i]
            optimizer: Optimizer = optimizers[agent_idx]
            # compute q-values
            qs: List[Tensor] = env.run(*actions)
            loss: Tensor = -qs[i]
            # backprop
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if episode == fix_inp_time:
            fix_inp = True

        if episode % 50 == 0:
            # computations
            state: Tensor = env.reset(fix_inp=True)
            actions: List[Tensor] = []
            for i in range(0, num_players):
                action_params: Tensor = ordered_agents[i](state, reward_distribution, player_tokens[i])
                actions.append(action_params)
            qs: List[Tensor] = env.run(*actions)
            # monitoring
            print("Step: {}".format(episode))
            print("Actions: {}".format(actions))
            print("Rewards: {}".format(qs))
            print("-------")


if __name__ == '__main__':
    main()
