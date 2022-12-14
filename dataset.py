# py
from abc import ABC
from typing import List, Tuple, Optional, Iterator
from time import time
from os.path import exists
from itertools import chain

# nn & rl
from torch import Tensor, eye, save, load
from torch.utils.data import Dataset, DataLoader, IterableDataset

# lib
from multi_env import MultiEnv
from nash import compute_nash_eq_b
from action_space import RestrictedActionSpace


def create_env() -> MultiEnv:
    num_players: int = 2
    env: MultiEnv = MultiEnv(num_players=num_players)
    env.action_space = RestrictedActionSpace()
    return env


class MicroIQND(IterableDataset, ABC):
    """
    Micro Iterable Quantum Nash-Equilibria Dataset
    """
    _iterable: List[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]

    def __init__(self, xs: List[Tuple[Tensor, Tensor, Tensor]], targets: List[Tensor]):
        super(MicroIQND, self).__init__()
        self._iterable = list(zip(xs, targets))

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)


def compute_max_dx_idx() -> int:
    ds_exists: bool = True
    max_ds_idx: int = 0
    while ds_exists:
        path = get_path(max_ds_idx)
        if exists(path):
            max_ds_idx += 1
            continue
        else:
            ds_exists = False
    return max_ds_idx


class IQND(IterableDataset, ABC):
    _iterable: List[MicroIQND]

    def __init__(self):
        max_ds_idx: int = compute_max_dx_idx()
        ds_iter: List[IterableDataset] = list()
        for idx in range(0, max_ds_idx):
            ds: IterableDataset = load(get_path(idx))
            ds_iter.append(ds)
        self._iterable = ds_iter

    def __iter__(self) -> Iterator:
        return chain(*self._iterable)

    def __len__(self) -> int:
        size: int = 0
        for ds in self._iterable:
            size += len(ds)
        return size


def create_samples(num_games: int, env: MultiEnv) -> Tuple[List[Tuple[Tensor, Tensor, Tensor]], List[Tensor]]:
    # instantiate variables
    player_tokens: Tensor = eye(env.num_players)
    # storage
    xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
    targets: List[Tensor] = list()

    # set dataset composition
    symmetric_perc: float = 0.8
    num_symmetric_games: int = int(num_games * symmetric_perc)

    while len(xs) < env.num_players * num_games:
        # different types of games
        if len(xs) < env.num_players * num_symmetric_games:
            env.generate_random_symmetric()
        else:
            env.generate_random()
        # compute nash equilibrium
        nash_eq: Optional[List[Tuple[Tensor, ...]]] = compute_nash_eq_b(env)
        if nash_eq:
            state: Tensor = env.reset(True)
            reward_distribution: Tensor = env.reward_distribution
            for idx in range(0, env.num_players):
                player_token: Tensor = player_tokens[idx]
                x: Tuple[Tensor, Tensor, Tensor] = (state, reward_distribution, player_token)
                target: Tensor = nash_eq[idx]
                xs.append(x)
                targets.append(target)
    return xs, targets


def create_micro_dataset(num_games: int = 100) -> Dataset:
    # configure env
    env = create_env()
    # make dataset
    xs, targets = create_samples(num_games=num_games, env=env)
    dataset: Dataset = MicroIQND(xs, targets)
    return dataset


def get_path(idx: int) -> str:
    return f"dataset/micro-datasets/micro-dataset-{idx}.pth"


def main():
    iqnd: Dataset = IQND()




if __name__ == '__main__':
    main()

# torch.utils.data.random_split()
