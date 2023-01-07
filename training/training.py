# py
from typing import Callable, List
from os.path import exists

# nn & rl
from torch import Tensor, tensor, save, load
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.nn.modules import HuberLoss

# lib
from base.utils import create_env
from base.env import Env
from base.transformer import Transformer
from dataset.dataset import QuantumTrainingDataset, MicroQuantumTrainingDataset


def validate(val_ds: DataLoader, loss_function: Callable[[Tensor, Tensor], Tensor], agent: Transformer) -> Tensor:
    val_losses: List[Tensor] = list()
    for x_batch, nash_batch in val_ds:
        action_batch: Tensor = agent(*x_batch)
        loss = loss_function(action_batch, nash_batch)
        val_losses.append(loss)
    return tensor(val_losses).mean(0)


def test(test_ds: QuantumTrainingDataset, agent: Transformer) -> Tensor:
    loss_function: Callable[[Tensor, Tensor], Tensor] = HuberLoss()
    test_losses: List[Tensor] = list()
    for x, nash in test_ds:
        action: Tensor = agent(*x)
        loss: Tensor = loss_function(action, nash)
        test_losses.append(loss)
    return tensor(test_losses).mean(0)


def train(num_episodes: int,
          agent: Transformer,
          optimizer: Optimizer,
          train_ds: DataLoader,
          val_ds: DataLoader) -> Transformer:
    loss_function: Callable[[Tensor, Tensor], Tensor] = HuberLoss()
    for episode in range(1, num_episodes + 1):
        train_losses: List[Tensor] = list()
        for i, (x_batch, nash_batch) in enumerate(train_ds):
            action_batch: Tensor = agent(*x_batch)
            loss: Tensor = loss_function(action_batch, nash_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
            if i % 50 == 0:
                # monitoring
                mean_val_loss: Tensor = validate(val_ds, loss_function, agent)
                mean_train_loss: Tensor = tensor(train_losses).mean(0)
                print(f"Episode: {episode} - Train Loss: {mean_train_loss} - Validation Loss: {mean_val_loss}")
        if episode % 5 == 0:
            # save model
            save_path: str = get_next_model_path()
            save(agent, save_path)
    return agent


def get_max_idx() -> int:
    is_max: bool = False
    idx: int = -1
    while not is_max:
        path: str = get_model_path(idx + 1)
        if exists(path):
            idx += 1
            continue
        else:
            return idx


def get_model_path(idx: int) -> str:
    return f"training/models/transformer-{idx}.pt"


def get_next_model_path() -> str:
    next_idx: int = get_max_idx() + 1
    return get_model_path(next_idx)


def main():
    # create environment
    env: Env = create_env()
    # define hyperparameters
    num_players: int = env.num_players
    num_actions: int = env.action_space.num_params
    num_encoder_layers: int = 4
    # initialize agent
    agent: Transformer = Transformer(
        num_players=num_players,
        num_actions=num_actions,
        num_encoder_layers=num_encoder_layers,
    )
    # initialize optimizer
    adam: Optimizer = Adam(params=agent.parameters())
    # load datasets
    batch_size: int = 64
    # train dataset
    qt_train_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0., end=0.8)
    train_data_loader: DataLoader = DataLoader(
        dataset=qt_train_ds,
        batch_size=batch_size,
    )
    # validation dataset
    qt_val_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0.8, end=0.9)
    val_data_loader: DataLoader = DataLoader(
        dataset=qt_val_ds,
        batch_size=batch_size,
    )
    # test dataset
    qt_test_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0.9, end=1.)

    # training
    num_episodes: int = 30
    agent = train(
        num_episodes=num_episodes,
        agent=agent,
        optimizer=adam,
        train_ds=train_data_loader,
        val_ds=val_data_loader,
    )

    # testing
    test_loss: Tensor = test(qt_test_ds, agent)
    print(f"Test Loss: {test_loss}")

    # save model
    save_path: str = get_next_model_path()
    save(agent, save_path)

    # load
    agent: Transformer = load(get_model_path(get_max_idx()))

    # testing after loading
    test_loss: Tensor = test(qt_test_ds, agent)
    print(f"Test Loss: {test_loss}")


if __name__ == '__main__':
    main()
