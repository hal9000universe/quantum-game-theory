# py
from abc import ABC
from typing import List, Tuple, Optional, Iterator, Callable
from os.path import exists
from itertools import chain

# nn & rl
from torch import Tensor, eye, save, load
from torch.utils.data import Dataset, IterableDataset

# lib
from base.utils import create_env
from base.env import Env
from base.nash import compute_nash_eq_b, is_nash


"""This file is all about datasets, creating them, storing them, 
accessing them and checking the validity of the data."""


# The following functions help to find paths to some data files
def get_mixed_mqt_path(idx: int) -> str:
    return f"dataset/mixed/quantum-training-datasets/quantum-training-dataset-{idx}.pth"


def get_random_mqt_path(idx: int) -> str:
    return f"dataset/random/quantum-training-datasets/quantum-training-dataset-{idx}.pth"


def get_symmetric_mqt_path(idx: int) -> str:
    return f"dataset/symmetric/quantum-training-datasets/quantum-training-dataset-{idx}.pth"


def get_mixed_gn_path(idx: int) -> str:
    return f"dataset/mixed/game-nash-datasets/game-nash-dataset-{idx}.pth"


def get_random_gn_path(idx: int) -> str:
    return f"dataset/random/game-nash-datasets/game-nash-dataset-{idx}.pth"


def get_symmetric_gn_path(idx: int) -> str:
    return f"dataset/symmetric/game-nash-datasets/game-nash-dataset-{idx}.pth"


class MicroQuantumTrainingDataset(IterableDataset, ABC):
    """The smallest dataset entity for training purposes
    The data is structured as follows:
    x = (state, reward distribution, player-token), y = nash-equilibrium[player]."""
    _iterable: List[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]

    def __init__(self, xs: List[Tuple[Tensor, Tensor, Tensor]], targets: List[Tensor]):
        super(MicroQuantumTrainingDataset, self).__init__()
        self._iterable = list(zip(xs, targets))

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)


class QuantumTrainingDataset(IterableDataset, ABC):
    """Concatenates many smaller MicroQuantumTrainingDatasets.
    Chunking the datasets potentially allows for lazy loading (which was NOT implemented here)."""
    _iterable: List[MicroQuantumTrainingDataset]

    def __init__(self, start: Optional[int | float] = None, end: Optional[int | float] = None,
                 path_fun: Callable[[int], str] = get_mixed_mqt_path):
        max_ds_idx: int = compute_max_ds_idx(path_fun)
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
            ds: IterableDataset = load(path_fun(idx))
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
    """The smallest entity of a game storage dataset.
    The data is structured as follows:
    x = reward-distribution, y = nash-equilibrium."""
    _iterable: List[Tensor]

    def __init__(self, games: List[Tensor], nash_eqs: List[Tensor]):
        self._iterable = list(zip(games, nash_eqs))

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)


class GameNashDataset(IterableDataset, ABC):
    """Concatenates many MicroGameNashDatasets.
    Chunking the datasets potentially allows for lazy loading (which was NOT implemented here)."""
    _iterable: List[MicroGameNashDataset]

    def __init__(self, start: Optional[int | float] = None, end: Optional[int | float] = None,
                 path_fun: Callable[[int], str] = get_mixed_gn_path):
        max_ds_idx: int = compute_max_ds_idx(path_fun)
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
            ds: IterableDataset = load(path_fun(idx))
            ds_iter.append(ds)
        self._iterable = ds_iter

    def __iter__(self) -> Iterator:
        return chain(*self._iterable)

    def __len__(self) -> int:
        size: int = 0
        for ds in self._iterable:
            size += len(ds)
        return size


# finds the highest index for a given type of micro-dataset
def compute_max_ds_idx(fun: Callable[[int], str]) -> int:
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


# generates mixed reward distribution type quantum games, solves them and creates a dataset
def create_mixed_game_nash_dataset(num_games: int = 100) -> Dataset:
    env: Env = create_env()
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


# generates random reward distribution type quantum games, solves them and stores them in a dataset.
def create_random_game_nash_dataset(num_games: int = 100) -> Dataset:
    env: Env = create_env()
    # storage
    xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
    targets: List[Tensor] = list()

    while len(xs) < num_games:
        # generate random reward distribution
        env.generate_random()
        # compute nash equilibrium
        nash_eq: Optional[Tensor] = compute_nash_eq_b(env)
        if isinstance(nash_eq, Tensor):
            xs.append(env.reward_distribution)
            targets.append(nash_eq)

    return MicroGameNashDataset(games=xs, nash_eqs=targets)


# generates symmetric reward distribution type quantum games, solves them and stores them in a dataset.
def create_symmetric_game_nash_dataset(num_games: int = 100) -> Dataset:
    env: Env = create_env()
    # storage
    xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
    targets: List[Tensor] = list()

    while len(xs) < num_games:
        # generate symmetric reward distribution
        env.generate_random_symmetric()
        # compute nash equilibrium
        nash_eq: Optional[Tensor] = compute_nash_eq_b(env)
        if isinstance(nash_eq, Tensor):
            xs.append(env.reward_distribution)
            targets.append(nash_eq)

    return MicroGameNashDataset(games=xs, nash_eqs=targets)


# creates one large game nash dataset (type mixed) by generating a large number of micro-datasets.
def construct_mixed_game_nash_dataset(size: int = 12500) -> GameNashDataset:
    micro_ds_size: int = 100
    num_micro_ds: int = size // micro_ds_size
    starting_index: int = compute_max_ds_idx(get_mixed_gn_path)
    for mds in range(starting_index, num_micro_ds):
        print(f"Current micro game-nash index: {mds}")
        micro_gnds: MicroGameNashDataset = create_mixed_game_nash_dataset(micro_ds_size)
        save(micro_gnds, get_mixed_gn_path(mds))
    return GameNashDataset(path_fun=get_mixed_gn_path)


# creates one large game nash dataset (type random) by generating a large number of micro-datasets.
def construct_random_game_nash_dataset(size: int = 12500) -> GameNashDataset:
    micro_ds_size: int = 100
    num_micro_ds: int = size // micro_ds_size
    starting_index: int = compute_max_ds_idx(get_random_gn_path)
    for mds in range(starting_index, num_micro_ds):
        print(f"Current micro game-nash index: {mds}")
        micro_gnds: MicroGameNashDataset = create_random_game_nash_dataset(micro_ds_size)
        save(micro_gnds, get_random_gn_path(mds))
    return GameNashDataset(path_fun=get_random_gn_path)


# creates one large game nash dataset (type symmetric) by generating a large number of micro-datasets.
def construct_symmetric_game_nash_dataset(size: int = 12500) -> GameNashDataset:
    micro_ds_size: int = 100
    num_micro_ds: int = size // micro_ds_size
    starting_index: int = compute_max_ds_idx(get_symmetric_gn_path)
    for mds in range(starting_index, num_micro_ds):
        print(f"Current micro game-nash dataset index: {mds}")
        micro_gnds: MicroGameNashDataset = create_symmetric_game_nash_dataset(micro_ds_size)
        save(micro_gnds, get_symmetric_gn_path(mds))
    return GameNashDataset(path_fun=get_random_gn_path)


# converts a large game nash dataset (type mixed) into a dataset for training mode.
def construct_mixed_training_dataset() -> QuantumTrainingDataset:
    env: Env = create_env()
    player_tokens: Tensor = eye(env.num_players)
    max_idx: int = compute_max_ds_idx(get_mixed_gn_path)
    for idx in range(0, max_idx):
        gn_ds: MicroGameNashDataset = load(get_mixed_gn_path(idx))
        xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
        ys: List[Tensor] = list()
        for game, solution in gn_ds:
            for player in range(0, env.num_players):
                x: Tuple[Tensor, Tensor, Tensor] = (env.reset(True), game, player_tokens[player])
                y: Tensor = solution[player]
                xs.append(x)
                ys.append(y)
        mqt_ds: MicroQuantumTrainingDataset = MicroQuantumTrainingDataset(xs, ys)
        save(mqt_ds, get_mixed_mqt_path(idx))
    qt_ds: QuantumTrainingDataset = QuantumTrainingDataset(path_fun=get_mixed_mqt_path)
    return qt_ds


# converts a large game nash dataset (type random) into a dataset for training mode.
def construct_random_training_dataset() -> QuantumTrainingDataset:
    env: Env = create_env()
    player_tokens: Tensor = eye(env.num_players)
    max_idx: int = compute_max_ds_idx(get_random_gn_path)
    for idx in range(0, max_idx):
        gn_ds: MicroGameNashDataset = load(get_random_gn_path(idx))
        xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
        ys: List[Tensor] = list()
        for game, solution in gn_ds:
            for player in range(0, env.num_players):
                x: Tuple[Tensor, Tensor, Tensor] = (env.reset(True), game, player_tokens[player])
                y: Tensor = solution[player]
                xs.append(x)
                ys.append(y)
        mqt_ds: MicroQuantumTrainingDataset = MicroQuantumTrainingDataset(xs, ys)
        save(mqt_ds, get_random_mqt_path(idx))
    qt_ds: QuantumTrainingDataset = QuantumTrainingDataset(path_fun=get_random_gn_path)
    return qt_ds


# converts a large game nash dataset (type symmetric) into a dataset for training mode.
def construct_symmetric_training_dataset() -> QuantumTrainingDataset:
    env: Env = create_env()
    player_tokens: Tensor = eye(env.num_players)
    max_idx: int = compute_max_ds_idx(get_symmetric_gn_path)
    for idx in range(0, max_idx):
        gn_ds: MicroGameNashDataset = load(get_symmetric_gn_path(idx))
        xs: List[Tuple[Tensor, Tensor, Tensor]] = list()
        ys: List[Tensor] = list()
        for game, solution in gn_ds:
            for player in range(0, env.num_players):
                x: Tuple[Tensor, Tensor, Tensor] = (env.reset(True), game, player_tokens[player])
                y: Tensor = solution[player]
                xs.append(x)
                ys.append(y)
        mqt_ds: MicroQuantumTrainingDataset = MicroQuantumTrainingDataset(xs, ys)
        save(mqt_ds, get_symmetric_mqt_path(idx))
    qt_ds: QuantumTrainingDataset = QuantumTrainingDataset(path_fun=get_symmetric_mqt_path)
    return qt_ds


# checks the correctness of a dataset's nash equilibria with
# higher discretization fidelity than the one used for creation
def check_dataset(path_fun: Callable[[int], str] = get_mixed_gn_path):
    # Correct: 12493, Total: 12600, action_space._num_steps = 24
    # Success rate: 12493 / 12600 = 0.9915079365079366.
    gn_ds: GameNashDataset = GameNashDataset(path_fun=path_fun)
    environment: Env = create_env()
    env.action_space.num_steps = 30

    num_correct: int = 0
    num_total: int = 0
    for i, (game, nash) in enumerate(gn_ds):
        environment.reward_distribution = game
        check: bool = is_nash(list(nash), environment)
        num_total += 1
        if check:
            num_correct += 1
        print(f"Game {i}: {check}")

    print(f"Out of {num_total} games, {num_correct} were solved correctly.")
    print(f"Success rate: {num_correct / num_total}")


# filters out wrong game - nash equilibrium pairs
def filter_dataset(path_fun: Callable[[int], str] = get_mixed_gn_path):
    gn_ds: GameNashDataset = GameNashDataset(path_fun=path_fun)
    environment: Env = create_env()
    environment.action_space.num_steps = 30

    verified_games: List[Tensor] = list()
    verified_nash: List[Tensor] = list()
    num_micro_datasets: int = 0
    for i, (game, nash) in enumerate(gn_ds, start=1):
        environment.reward_distribution = game
        if is_nash(list(nash), environment):
            verified_games.append(game)
            verified_nash.append(nash)
        if len(verified_games) == 100:
            micro_gn_ds: MicroGameNashDataset = MicroGameNashDataset(verified_games, verified_nash)
            save(micro_gn_ds, path_fun(num_micro_datasets))
            verified_games = list()
            verified_nash = list()
            num_micro_datasets += 1
        print(f"Game: {i} - Number of Micro Datasets: {num_micro_datasets} - len(verified) = {len(verified_games)}")

    print(f"Final filtered Dataset: {num_micro_datasets}")
