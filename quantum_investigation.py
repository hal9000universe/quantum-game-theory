# check if env works as expected
from torch import tensor, complex64, kron, matrix_exp, exp, sin, cos, arange
from math import pi


# rotation_operator
def rotation_operator(theta, phi):
    return tensor([[exp(1j * phi) * cos(theta / 2), sin(theta / 2)],
                   [-sin(theta / 2), exp(-1j * phi) * cos(theta / 2)]])


def main(theta_noise=tensor(0.), phi_noise=tensor(0.)):
    # prepare ground state
    ground_state = tensor([1., 0., 0., 0.], dtype=complex64)

    # construct gate to prepare initial state
    D = tensor([[0., 1.], [1., 0.]], dtype=complex64)
    gamma = pi / 2
    J = matrix_exp(-1j * gamma * kron(D, D) / 2)

    # prepare initial state
    initial_state = J @ ground_state

    # strategy representation
    theta = tensor(0.)
    phi = tensor(pi / 2)
    Q = rotation_operator(theta, phi)
    Q2 = rotation_operator(theta + theta_noise, phi + phi_noise)

    # apply strategy
    altered_state = kron(Q, Q2) @ initial_state

    # final state
    final_state = J.transpose(0, 1).conj() @ altered_state
    print(final_state)

    # calculate rewards
    rewards = tensor([[3., 3.], [5., 0.], [0., 5.], [1., 1.]])
    optimal_strat_reward = (rewards[:, 0] * final_state.abs().square()).sum()
    bob_strat_reward = (rewards[:, 1] * final_state.abs().square()).sum()
    optimal = optimal_strat_reward >= bob_strat_reward
    return optimal_strat_reward, bob_strat_reward, optimal


if __name__ == '__main__':
    theta_noise_range = arange(0., pi, step=0.1)
    phi_noise_range = arange(0., pi / 2, step=0.1)
    for theta_step in theta_noise_range:
        for phi_step in phi_noise_range:
            opt, sub, is_opt = main(theta_step, phi_step)
            print(opt, sub, is_opt.item())
