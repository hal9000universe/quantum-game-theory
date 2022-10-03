from torch import tensor, relu, Tensor, cat
from torch.nn import Module, Linear
from torch.optim import Adam
from itertools import chain


class Env:

    def __init__(self):
        self._inp = tensor([1., 1.])
        self._goal = tensor(10.)

    def reset(self):
        return self._inp

    def step(self, guess1, guess2):
        reward2 = - (guess1 / guess2 - self._goal).square()
        reward1 = - (guess1 - self._goal).square()
        return reward1, reward2


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self._lin1 = Linear(2, 10)
        self._lin2 = Linear(10, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = relu(x)
        x = self._lin2(x)
        return x


class QNet(Module):

    def __init__(self):
        super(QNet, self).__init__()
        self._lin1 = Linear(3, 10)
        self._lin2 = Linear(10, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._lin1(x)
        x = relu(x)
        x = self._lin2(x)
        return x


if __name__ == '__main__':
    env = Env()
    alice = Net()
    q_alice = QNet()
    bob = Net()
    q_bob = QNet()

    q_optimizer = Adam(params=chain(q_alice.parameters(), q_bob.parameters()))
    ac_optimizer = Adam(params=chain(alice.parameters(), bob.parameters()))

    for step in range(100000):
        inp = env.reset()

        ac_alice = alice(inp)
        ac_bob = bob(inp)

        rew_alice, rew_bob = env.step(ac_alice, ac_bob)

        # update q-networks
        q_inp_alice = cat((inp, ac_alice))
        q_inp_bob = cat((inp, ac_bob))

        q_val_alice = q_alice(q_inp_alice)
        q_val_bob = q_bob(q_inp_bob)

        q_loss_alice = (q_val_alice - rew_alice).square()
        q_loss_bob = (q_val_bob - rew_bob).square()
        q_loss = q_loss_alice + q_loss_bob

        q_optimizer.zero_grad()
        q_loss.backward()  # alternatively retain_graph=True
        q_optimizer.step()

        # update agents
        ac_alice = alice(inp)
        ac_bob = bob(inp)

        q_inp_alice = cat((inp, ac_alice))
        q_inp_bob = cat((inp, ac_bob))

        q_val_alice = q_alice(q_inp_alice)
        q_val_bob = q_bob(q_inp_bob)

        alice_loss = -q_val_alice
        bob_loss = -q_val_bob
        pg_loss = alice_loss + bob_loss

        ac_optimizer.zero_grad()
        pg_loss.backward()
        ac_optimizer.step()

        if step % 1000 == 0:
            reward = rew_bob + rew_alice
            print(reward.item())
            print(ac_alice.item(), ac_bob.item())
            print('---------')
