# py
from math import pi
from typing import Optional, Tuple, Callable, List

# nn & rl
from torch import tensor, Tensor, cat, kron, real, ones, complex64, float32, allclose
from torch import relu, sigmoid, exp, sin, cos, matrix_exp
from torch.nn import Module, Linear
from torch.nn.init import kaiming_normal_
from torch.optim import Adam
from torch.distributions import Uniform, Distribution

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import QuantumSystem, Operator
from action_space import RestrictedActionSpace, ActionSpace


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


def create_circuit() -> Callable:
    dev: Device = device('default.qubit', wires=2)
    all_wires: List[int] = [0, 1]

    gamma: float = pi / 2
    D: Tensor = tensor([[0., 1.],
                        [-1., 0.]], dtype=complex64)
    J = matrix_exp(-1j * gamma * kron(D, D) / 2)

    @qnode(device=dev, interface='torch')
    def circuit(U: Tensor) -> Tensor:
        QubitUnitary(J, wires=all_wires)
        QubitUnitary(U, wires=all_wires)
        QubitUnitary(J.adjoint(), wires=all_wires)
        return probs(wires=all_wires)

    return circuit


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

    @property
    def reward_distribution(self):
        return self._rewards

    @property
    def action_space(self):
        return RestrictedActionSpace()

    @property
    def _ground_state(self) -> QuantumSystem:
        return QuantumSystem(num_qubits=2)

    def reset(self, fix_inp: bool = False) -> Tensor:
        """returns a random input"""
        if fix_inp:
            return 0.1 * ones((4,)).type(complex64)
        else:
            rand_inp = self._uniform.sample((4,)).type(complex64)
            return rand_inp

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

    def quantum_step(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        # create circuit
        circuit: Callable = create_circuit()
        # create rotation operators given by a1 and a2
        u1: Operator = rotation_operator(a1)
        u2: Operator = rotation_operator(a2)
        U: Tensor = (u1 + u2).mat
        # run quantum circuit
        probabilities: Tensor = circuit(U)
        # calculate q-values
        q1: Tensor = (self._rewards[:, 0] * probabilities).sum()
        q2: Tensor = (self._rewards[:, 1] * probabilities).sum()
        # return q-values
        return q1, q2


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
        self._lin1 = Linear(4, 128, dtype=complex64)
        self._lin2 = Linear(128, 128, dtype=float32)
        self._lin3 = Linear(128, 128, dtype=float32)
        self._lin4 = Linear(128, 2, dtype=float32)
        self._scaling = tensor([pi, pi / 2])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(m):
        """
        Initializes weights using the kaiming-normal distribution and sets biases to zero.
        """
        if isinstance(m, Linear):
            # manual_seed(2000)  # ensures reproducibility
            m.weight = kaiming_normal_(m.weight)
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
    env: Env = Env()
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()
    al_op = Adam(params=alice.parameters())
    bo_op = Adam(params=bob.parameters())

    episodes: int = 1000
    fix_inp: bool = True
    fix_inp_time: int = int(episodes * 0.6)

    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)

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

        # compute actions
        ac_bob = bob(state)
        ac_alice = alice(state)

        # compute q-values
        q_alice, q_bob = env.step(ac_alice, ac_bob)

        # define loss
        bob_loss: Tensor = -q_bob

        # optimize bob
        bo_op.zero_grad()
        bob_loss.backward()
        bo_op.step()

        if step % 5000 == 0:  # lower for better monitoring
            print('step: {}'.format(step))
            print(q_alice.item(), q_bob.item())
            print(ac_alice, ac_bob)
            print('---------')

        if step == fix_inp_time:
            fix_inp = True

    state = env.reset()
    ac_al = alice(state)
    ac_bo = bob(state)
    reward_a, reward_b = env.step(ac_al, ac_bo)
    print('-----')
    print('actions after training: ')
    print('alice: {}'.format(ac_al))
    print('bob: {}'.format(ac_bo))
    print('reward after training: {} {}'.format(reward_a.item(), reward_b.item()))
    return ac_al, ac_bo


def qmain():
    # initialize base classes
    env: Env = Env()
    alice: Module = ComplexNetwork()
    bob: Module = ComplexNetwork()
    al_op = Adam(params=alice.parameters())
    bo_op = Adam(params=bob.parameters())

    episodes: int = 1000
    fix_inp: bool = False
    fix_inp_time: int = int(episodes * 0.6)

    # loop over episodes
    for step in range(1, episodes):
        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_alice = alice(state)
        ac_bob = bob(state)

        # compute q-values
        q_alice, q_bob = env.quantum_step(ac_alice, ac_bob)

        # define loss
        alice_loss: Tensor = -q_alice

        # optimize alice
        al_op.zero_grad()
        alice_loss.backward()
        al_op.step()

        # get state from environment
        state = env.reset(fix_inp=fix_inp)

        # compute actions
        ac_bob = bob(state)
        ac_alice = alice(state)

        # compute q-values
        q_alice, q_bob = env.quantum_step(ac_alice, ac_bob)

        # define loss
        bob_loss: Tensor = -q_bob

        # optimize bob
        bo_op.zero_grad()
        bob_loss.backward()
        bo_op.step()

        if step % 50 == 0:
            print('step: {}'.format(step))
            print(q_alice.item(), q_bob.item())
            print(ac_alice, ac_bob)
            print('---------')

        if step == fix_inp_time:
            fix_inp = True

    state = env.reset()
    ac_al = alice(state)
    ac_bo = bob(state)
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
    times: int = 30
    for time in range(times):
        final_params1, final_params2 = main()
        if check(final_params1) and check(final_params2):
            print('training successful ...')
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


def quantum_success_simulation():
    nums: int = 0
    times: int = 10
    for time in range(times):
        final_params1, final_params2 = qmain()
        if check(final_params1) and check(final_params2):
            print('training successful ...')
            nums += 1
    success_rate: float = nums / times
    print('success rate: {}'.format(success_rate))


if __name__ == '__main__':
    sim_success_evaluation()
    # result (fluctuating input): 1. success rate
    # result (fixed input): 0.2 success rate
