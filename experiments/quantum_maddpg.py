# py
from typing import Callable

# nn & rl
from torch import Tensor, load

# lib
from base.utils import calc_dist
from base.general import training_framework
from base.action_space import RestrictedActionSpace
from dataset.dataset import MicroGameNashDataset


def test_algorithm(load_model: bool = False, noisy_inputs: bool = False, noisy_actions: bool = False) -> float:
    ds = load("dataset/game-nash-datasets/game-nash-dataset-125.pth")
    parametrization: Callable[[Tensor], Tensor] = RestrictedActionSpace().operator

    num_games: int = len(ds)
    num_successes: int = 0

    for i, (reward_distribution, nash_eq) in enumerate(ds, start=1):
        actions: Tensor = training_framework(
            reward_distribution=reward_distribution,
            load_model=load_model,
            noisy_inputs=noisy_inputs,
            noisy_actions=noisy_actions,
        )
        dist: Tensor = calc_dist(nash_eq, actions, parametrization)
        if dist < 0.2:
            num_successes += 1
        print(f"Game: {i} - Dist: {dist} - Successes: {num_successes}")

    return num_successes / num_games


def no_exploration_test():
    # Success rate: 0.43 - 0.46
    accuracy: float = test_algorithm(
        load_model=False,
        noisy_inputs=False,
        noisy_actions=False,
    )
    print(f"No Exploration Accuracy: {accuracy}")


def pretrained_no_exploration_test():
    # Success rate: 0.82 - 0.87
    accuracy: float = test_algorithm(
        load_model=True,
        noisy_inputs=False,
        noisy_actions=False,
    )
    print(f"Pretrained No Exploration Accuracy: {accuracy}")


def noisy_inputs_test():
    # Success rate: 0.43 - 0.45
    accuracy: float = test_algorithm(
        load_model=False,
        noisy_inputs=True,
        noisy_actions=False,
    )
    print(f"Noisy Inputs Accuracy: {accuracy}")


def pretrained_noisy_inputs_test():
    # Success rate: 0.85 - 0.88
    accuracy: float = test_algorithm(
        load_model=True,
        noisy_inputs=True,
        noisy_actions=False,
    )
    print(f"Pretrained Noisy Inputs Accuracy: {accuracy}")


def noisy_actions_test():
    # Success rate: 0.41 - 0.47
    accuracy: float = test_algorithm(
        load_model=False,
        noisy_inputs=False,
        noisy_actions=True,
    )
    print(f"Noisy Actions Accuracy: {accuracy}")


def pretrained_noisy_actions_test():
    # Success rate: 0.88 - 0.89
    accuracy: float = test_algorithm(
        load_model=True,
        noisy_inputs=False,
        noisy_actions=True,
    )
    print(f"Pretrained Noisy Actions Accuracy: {accuracy}")


if __name__ == '__main__':
    no_exploration_test()
    pretrained_no_exploration_test()
    noisy_inputs_test()
    pretrained_noisy_inputs_test()
    noisy_actions_test()
    pretrained_noisy_actions_test()
