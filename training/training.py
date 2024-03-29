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
from dataset.dataset import QuantumTrainingDataset, MicroQuantumTrainingDataset, get_mixed_mqt_path, get_random_mqt_path, get_symmetric_mqt_path


"""This file is about pretraining a transformer on a quantum game dataset."""


# The following functions help to find paths of pretrained models
def get_mixed_model_path(idx: int) -> str:
    return f"training/models/mixed/transformer-{idx}.pt"


def get_random_model_path(idx: int) -> str:
    return f"training/models/random/transformer-{idx}.pt"


def get_symmetric_model_path(idx: int) -> str:
    return f"training/models/symmetric/transformer-{idx}.pt"


# maps a function for the path of some data to a function for the path of a model trained on that data
def path_fun_map(data_path_fun: Callable[[int], str]) -> Callable[[int], str]:
    path_fun_name: str = data_path_fun.__name__
    if path_fun_name == "get_mixed_mqt_path":
        return get_mixed_model_path
    elif path_fun_name == "get_random_mqt_path":
        return get_random_model_path
    elif path_fun_name == "get_symmetric_mqt_path":
        return get_symmetric_model_path
    elif path_fun_name == "get_mixed_gn_path":
        return get_mixed_model_path
    elif path_fun_name == "get_random_gn_path":
        return get_random_model_path
    elif path_fun_name == "get_symmetric_gn_path":
        return get_symmetric_model_path
    else:
        print("Unexpected behaviour occurred in: path_fun_map.")
        exit()


# calculates the index of the most trained model depending on the kind of data it was trained on
def get_max_idx(path_fun: Callable[[int], str] = get_mixed_model_path) -> int:
    is_max: bool = False
    idx: int = 0
    while not is_max:
        path: str = path_fun(idx + 1)
        if exists(path):
            idx += 1
            continue
        else:
            return idx


# computes the validation loss for a model
def validate(val_ds: DataLoader, loss_function: Callable[[Tensor, Tensor], Tensor], agent: Transformer) -> Tensor:
    val_losses: List[Tensor] = list()
    for x_batch, nash_batch in val_ds:
        action_batch: Tensor = agent(*x_batch)
        loss = loss_function(action_batch, nash_batch)
        val_losses.append(loss)
    return tensor(val_losses).mean(0)


# computes the test loss for a model
def test(test_ds: QuantumTrainingDataset, agent: Transformer) -> Tensor:
    loss_function: Callable[[Tensor, Tensor], Tensor] = HuberLoss()
    test_losses: List[Tensor] = list()
    for x, nash in test_ds:
        action: Tensor = agent(*x)
        loss: Tensor = loss_function(action, nash)
        test_losses.append(loss)
    return tensor(test_losses).mean(0)


def train(num_epochs: int,
          agent: Transformer,
          optimizer: Optimizer,
          train_ds: DataLoader,
          val_ds: DataLoader,
          path_fun: Callable[[int], str] = get_mixed_model_path) -> Transformer:

    # initialize huber loss function
    loss_function: Callable[[Tensor, Tensor], Tensor] = HuberLoss()

    # loop over epochs
    for epoch in range(1, num_epochs + 1):

        # save model
        if epoch % 1 == 0:
            save(agent, path_fun(epoch))

        train_losses: List[Tensor] = list()  # variable to store losses
        for i, (x_batch, nash_batch) in enumerate(train_ds):
            action_batch: Tensor = agent(*x_batch)  # process data and get actions
            loss: Tensor = loss_function(action_batch, nash_batch)  # compute distance to nash equilibrium
            optimizer.zero_grad()  # delete gradients
            loss.backward()  # back-propagate gradients
            optimizer.step()  # update weights
            train_losses.append(loss)  # store losses

            # monitoring
            if i % 100 == 0:
                mean_val_loss: Tensor = validate(val_ds, loss_function, agent)
                mean_train_loss: Tensor = tensor(train_losses).mean(0)
                print(f"Episode: {epoch}.{i} - Train Loss: {mean_train_loss} - Validation Loss: {mean_val_loss}")

    # save trained agent
    save(agent, path_fun(get_max_idx(path_fun)))

    # return trained agent
    return agent


# trains an agent on a given dataset (identified by its path)
def main(data_path_fun: Callable[[int], str] = get_mixed_model_path):
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
    model_path_fun: Callable[[int], str] = path_fun_map(data_path_fun)
    qt_train_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0., end=0.8, path_fun=data_path_fun)
    train_data_loader: DataLoader = DataLoader(
        dataset=qt_train_ds,
        batch_size=batch_size,
    )
    # validation dataset
    qt_val_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0.8, end=0.9, path_fun=data_path_fun)
    val_data_loader: DataLoader = DataLoader(
        dataset=qt_val_ds,
        batch_size=batch_size,
    )
    # test dataset
    qt_test_ds: QuantumTrainingDataset = QuantumTrainingDataset(start=0.9, end=1., path_fun=data_path_fun)

    # training
    num_epochs: int = 40
    agent = train(
        num_epochs=num_epochs,
        agent=agent,
        optimizer=adam,
        train_ds=train_data_loader,
        val_ds=val_data_loader,
        path_fun=model_path_fun,
    )

    # testing
    test_loss: Tensor = test(qt_test_ds, agent)
    print(f"Test Loss: {test_loss}")

    # load
    agent: Transformer = load(model_path_fun(get_max_idx(model_path_fun)))

    # testing after loading
    test_loss: Tensor = test(qt_test_ds, agent)
    print(f"Test Loss: {test_loss}")


if __name__ == '__main__':
    main(data_path_fun=get_symmetric_mqt_path)
