# py
from abc import ABC
from typing import List, Tuple, Optional, Iterator, Callable
from os.path import exists
from itertools import chain

# nn & rl
from torch import Tensor, eye, save, load
from torch.utils.data import Dataset, IterableDataset, random_split

# lib
from base.utils import create_env
from base.multi_env import MultiEnv
from base.nash import compute_nash_eq_b


# LazyDataset (instead of allocating self._iterable, load self._iterable from file!)
class MicroQuantumTrainingDataset(IterableDataset, ABC):
    """
    Micro Iterable Quantum Nash-Equilibria Dataset
    """
    _iterable: List[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]

    def __init__(self, xs: List[Tuple[Tensor, Tensor, Tensor]], targets: List[Tensor]):
        super(MicroQuantumTrainingDataset, self).__init__()
        self._iterable = list(zip(xs, targets))

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)


class QuantumTrainingDataset(IterableDataset, ABC):
    _iterable: List[MicroQuantumTrainingDataset]

    def __init__(self, start: Optional[int | float] = None, end: Optional[int | float] = None):
        max_ds_idx: int = compute_max_ds_idx(get_mqt_path)
        ds_iter: List[IterableDataset] = list()
        if not start:
            start = 0
        if not end:
            end = max_ds_idx
        if isinstance(start, float):
            assert start < 1.
            start: int = (max_ds_idx * start).__int__()
        if isinstance(end, float):
            assert end <= 1.
            end: int = (max_ds_idx * end).__int__()
        for idx in range(start, end):
            ds: IterableDataset = load(get_mqt_path(idx))
            ds_iter.append(ds)
        self._iterable = ds_iter

    def __iter__(self) -> Iterator:
        return chain(*self._iterable)

    def __len__(self) -> int:
        size: int = 0
        for ds in self._iterable:
            size += len(ds)
        return size


class MicroGameNashDataset(IterableDataset, ABC):
    _iterable: List[Tensor]

    def __init__(self, games: List[Tensor], nash_eqs: List[Tensor]):
        self._iterable = list(zip(games, nash_eqs))

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)


class GameNashDataset(IterableDataset, ABC):
    _iterable: List[MicroGameNashDataset]

    def __init__(self):
        max_ds_idx: int = compute_max_ds_idx(get_gn_path)
        ds_iter: List[IterableDataset] = list()
        for idx in range(0, max_ds_idx):
            ds: IterableDataset = load(get_gn_path(idx))
            ds_iter.append(ds)
        self._iterable = ds_iter

    def __iter__(self) -> Iterator:
        return chain(*self._iterable)

    def __len__(self) -> int:
        size: int = 0
        for ds in self._iterable:
            size += len(ds)
        return size


def compute_max_ds_idx(fun: Callable) -> int:
    ds_exists: bool = True
    max_ds_idx: int = 0
    while ds_exists:
        path = fun(max_ds_idx)
        if exists(path):
            max_ds_idx += 1
            continue
        else:
            ds_exists = False
    return max_ds_idx


def create_game_nash_dataset(num_games: int = 100) -> Dataset:
    env: MultiEnv = create_env()
    # storage
    xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
    targets: List[Tensor] = list()

    # set dataset composition
    symmetric_perc: float = 0.8
    num_symmetric_games: int = int(num_games * symmetric_perc)

    while len(xs) < num_games:
        # different types of games
        if len(xs) < num_symmetric_games:
            env.generate_random_symmetric()
        else:
            env.generate_random()
        # compute nash equilibrium
        nash_eq: Optional[Tensor] = compute_nash_eq_b(env)
        if isinstance(nash_eq, Tensor):
            xs.append(env.reward_distribution)
            targets.append(nash_eq)

    return MicroGameNashDataset(games=xs, nash_eqs=targets)


def construct_training_dataset() -> QuantumTrainingDataset:
    env: MultiEnv = create_env()
    player_tokens: Tensor = eye(env.num_players)
    max_idx: int = compute_max_ds_idx(get_gn_path)
    for idx in range(0, max_idx):
        gn_ds: MicroGameNashDataset = load(get_gn_path(idx))
        xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
        ys: List[Tensor] = list()
        for game, solution in gn_ds:
            for player in range(0, env.num_players):
                x: Tuple[Tensor, Tensor, Tensor] = (env.reset(True), game, player_tokens[player])
                y: Tensor = solution[player]
                xs.append(x)
                ys.append(y)
        mqt_ds: MicroQuantumTrainingDataset = MicroQuantumTrainingDataset(xs, ys)
        save(mqt_ds, get_mqt_path(idx))
    qt_ds: QuantumTrainingDataset = QuantumTrainingDataset()
    return qt_ds


def get_mqt_path(idx: int) -> str:
    return f"dataset/quantum-training-datasets/quantum-training-dataset-{idx}.pth"


def get_gn_path(idx: int) -> str:
    return f"dataset/game-nash-datasets/game-nash-dataset-{idx}.pth"


def split_dataset(ds: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    ratio: List[float] = [0.8, 0.1, 0.1]
    train_ds, val_ds, test_ds = random_split(dataset=ds, lengths=ratio)
    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    qt_dataset = QuantumTrainingDataset(end=1.)
    print(len(qt_dataset))
    print("Done!")


# torch.utils.data.random_split()