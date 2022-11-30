# py
from math import pi
from typing import Optional, Tuple, Callable, List

# nn & rl
from torch import tensor, Tensor, cat, kron, real, ones, complex64, float32, allclose, eye
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam
from torch.distributions import Uniform, Distribution

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import QuantumSystem, Operator
from transformer import Transformer


def rotation_operator(params: Tensor) -> Operator:
    """
    computes and returns a complex rotation matrix given by the angles in params.
    """
    theta, phi = params
    # calculate entries
    a: Tensor = exp(1j * phi) * cos(theta / 2)
    b: Tensor = sin(theta / 2)
    c: Tensor = -b
    d: Tensor = exp(-1j * phi) * cos(theta / 2)
    # construct the rows of the rotation matrix
    r1: Tensor = cat((a.view(1), b.view(1)))
    r2: Tensor = cat((c.view(1), d.view(1)))
    # build and return the rotation matrix
    rot: Tensor = cat((r1, r2)).view(2, 2)
    return Operator(mat=rot)


class Env:
    _state: Optional[QuantumSystem]
    _J: Operator
    _rewards = Tensor
    _uniform: Distribution

    def __init__(self):
        """
        initializes Env class:
         Env implements the quantum circuit described in
         Quantum games and quantum strategies
         by Eisert et al. 2020.
        """
        # state variables
        self._state = None
        # operators
        gamma: float = pi / 2
        D: Tensor = tensor([[0., 1.],
                            [-1., 0.]], dtype=complex64)
        mat = matrix_exp(-1j * gamma * kron(D, D) / 2)
        self._J = Operator(mat=mat)
        # reward distribution
        self._rewards = tensor([[3., 3.], [0., 5.], [5., 0.], [1., 1.]])
        # create uniform input distribution
        self._uniform = Uniform(-0.25, 0.25)
        self._observation = (self._J @ self._ground_state).state

    @property
    def reward_distribution(self) -> Tensor:
        return self._rewards

    @property
    def _ground_state(self) -> QuantumSystem:
        return QuantumSystem(num_qubits=2)

    def reset(self, fix_inp: bool = False) -> Tensor:
        """returns a random input"""
        if fix_inp:
            return self._observation
        else:
            rand_inp = self._uniform.sample((4,)).type(complex64)
            return self._observation + rand_inp

    def step(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        calculates and returns the q-values for alice and bob given their respective actions.
        """
        # prepare initial state
        self._state = self._J @ self._ground_state
        # create rotation operators given by a1 and a2
        rot1: Operator = rotation_operator(a1)
        rot2: Operator = rotation_operator(a2)
        # apply tensor product to create an operator which acts on the 2-qubit system
        op: Operator = rot1 + rot2
        # apply rotation operators to the quantum system
        self._state = op @ self._state
        # apply adjoint of J
        self._state = self._J.adjoint @ self._state
        # calculate q-values
        q1: Tensor = (self._rewards[:, 0] * self._state.probs).sum()
        q2: Tensor = (self._rewards[:, 1] * self._state.probs).sum()
        # return q-values
        return q1, q2


def main():
    # initialize base classes
    env: Env = Env()
    alice: Module = Transformer(num_players=2, num_encoder_layers=2, num_actions=2)
    bob: Module = Transformer(num_players=2, num_encoder_layers=2, num_actions=2)
    al_op = Adam(params=alice.parameters())
    bo_op = Adam(params=bob.parameters())
    player_tokens = eye(2)

    episodes: int = 500
    fix_inp: bool = False
    fix_inp_time: int = int(episodes * 0.6)

    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)
        inp_alice = (state, env.reward_distribution, player_tokens[0])
        inp_bob = (state, env.reward_distribution, player_tokens[1])

        # compute actions
        ac_alice = alice(*inp_alice)
        ac_bob = bob(*inp_bob)

        # compute q-values
        q_alice, q_bob = env.step(ac_alice, ac_bob)

        # define loss
        alice_loss: Tensor = -q_alice

        # optimize alice
        al_op.zero_grad()
        alice_loss.backward()
        al_op.step()

        # get state from environment
        state = env.reset(fix_inp=fix_inp)
        inp_alice = (state, env.reward_distribution, player_tokens[0])
        inp_bob = (state, env.reward_distribution, player_tokens[1])

        # compute actions
        ac_alice = alice(*inp_alice)
        ac_bob = bob(*inp_bob)

        # compute q-values
        q_alice, q_bob = env.step(ac_alice, ac_bob)

        # define loss
        bob_loss: Tensor = -q_bob

        # optimize bob
        bo_op.zero_grad()
        bob_loss.backward()
        bo_op.step()

        if step % 5000 == 0:
            print('step: {}'.format(step))
            print(q_alice.item(), q_bob.item())
            print(ac_alice, ac_bob)
            print('---------')

        if step == fix_inp_time:
            fix_inp = True

    state = env.reset()
    inp_alice = (state, env.reward_distribution, player_tokens[0])
    inp_bob = (state, env.reward_distribution, player_tokens[1])
    ac_al = alice(*inp_alice)
    ac_bo = bob(*inp_bob)
    reward_a, reward_b = env.step(ac_al, ac_bo)
    print('-----')
    print('actions after training: ')
    print('alice: {}'.format(ac_al))
    print('bob: {}'.format(ac_bo))
    print('reward after training: {} {}'.format(reward_a.item(), reward_b.item()))
    return ac_al, ac_bo


def check(final_params: Tensor) -> bool:
    learned: bool = allclose(final_params, tensor([0., pi / 2]),
                             rtol=0.05,
                             atol=0.05)
    return learned


def sim_success_evaluation():
    nums: int = 0
    times: int = 20
    for time in range(times):
        final_params1, final_params2 = main()
        if check(final_params1) and check(final_params2):
            print('training successful ...')
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


if __name__ == '__main__':
    sim_success_evaluation()
    # result: 1. success rate
