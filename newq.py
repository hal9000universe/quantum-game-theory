# py
from math import pi
from itertools import chain
from typing import Optional, Tuple, Callable, Dict, List
from random import uniform

# nn & rl
from torch import tensor, Tensor, cat, kron, full, real, zeros, manual_seed, allclose
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch import float32, complex64
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Optimizer
from torch.distributions import Uniform, Distribution
from bayes_opt import BayesianOptimization, UtilityFunction

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import QuantumSystem, Ops, Operator
from env import QuantumSuperiorityGame


class ComplexNetwork(Module):
    _lin1: Linear
    _lin2: Linear
    _lin3: Linear
    _lin4: Linear
    _scaling: Tensor

    def __init__(self):
        super(ComplexNetwork, self).__init__()
        """
        initializes ComplexNetwork class.
         ComplexNetwork implements a feed-forward neural network which is capable of handling 
         4-dimensional complex inputs. 
        """
        self._lin1 = Linear(8, 128, dtype=complex64)
        self._lin2 = Linear(128, 128, dtype=float32)
        self._lin3 = Linear(128, 128, dtype=float32)
        self._lin4 = Linear(128, 3, dtype=float32)
        self._scaling = tensor([pi, pi, pi])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        """
        Initializes weights using the kaiming-normal distribution and sets weights to zero.
        """
        if isinstance(m, Linear):
            # manual_seed(2000)  # ensures reproducibility
            kaiming_normal_(m.weight)
            m.bias.data.fill_(0.)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = real(x)
        x = self._lin2(x)
        x = self._lin3(x)
        x = self._lin4(x)
        x = self._scaling * sigmoid(x)
        return x


def main():
    # initialize base classes
    env: Env = QuantumSuperiorityGame()
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()
    charlie: Module = ComplexNetwork()
    al_op: Optimizer = Adam(params=alice.parameters())
    bo_op: Optimizer = Adam(params=bob.parameters())
    ch_op: Optimizer = Adam(params=charlie.parameters())

    fix_inp: bool = False
    episodes: int = 50000
    fix_inp_time: int = int(episodes * 0.6)

    # loop over episodes
    for step in range(episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)
        ac_charlie = charlie(state)

        # compute q-values
        q_alice, q_bob, q_charlie = env.step(ac_alice, ac_bob, ac_charlie)

        # define loss
        alice_loss: Tensor = -q_alice

        # optimize alice
        al_op.zero_grad()
        alice_loss.backward()
        al_op.step()

        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)
        ac_charlie = charlie(state)

        # compute q-values
        q_alice, q_bob, q_charlie = env.step(ac_alice, ac_bob, ac_charlie)

        # define loss
        bob_loss: Tensor = -q_bob

        # optimize bob
        bo_op.zero_grad()
        bob_loss.backward()
        bo_op.step()

        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)
        ac_charlie = charlie(state)

        # compute q-values
        q_alice, q_bob, q_charlie = env.step(ac_alice, ac_bob, ac_charlie)

        # define loss
        charlie_loss: Tensor = -q_charlie

        # optimize bob
        ch_op.zero_grad()
        charlie_loss.backward()
        ch_op.step()

        if step % 200 == 0:
            print('step: {}'.format(step))
            print(q_alice.item(), q_bob.item(), q_charlie.item())
            print(ac_alice, ac_bob, ac_charlie)
            print('---------')

        if step == fix_inp_time:
            fix_inp = True

    state = env.reset(fix_inp=True)
    ac_al = alice(state)
    ac_bo = bob(state)
    ac_ch = charlie(state)
    reward_a, reward_b, reward_c = env.step(ac_al, ac_bo, ac_ch)
    print('actions after training: ')
    print('alice: {}'.format(ac_al))
    print('bob: {}'.format(ac_bo))
    print('charlie: {}'.format(ac_ch))
    print('reward after training: {} {} {}'.format(reward_a.item(), reward_b.item(), reward_c.item()))
    return ac_al, ac_bo, ac_ch


def check(final_params: Tensor) -> bool:
    learned: bool = allclose(final_params, tensor([0., pi / 2]),
                             rtol=0.1,
                             atol=0.1)
    return learned


def success_evaluation():
    nums: int = 0
    times: int = 20
    for time in range(times):
        final_params1, final_params2 = main()
        if check(final_params1) and check(final_params2):
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


if __name__ == '__main__':
    main()
