# nn & rl
from torch import tensor, Tensor
from torch.optim import Adam, Optimizer

# quantum
from pennylane import qnode, QubitUnitary, probs, device, Device

# lib
from quantum import Ops
from qmain import ComplexNetwork, Env


if __name__ == '__main__':
    # define variables
    alice = ComplexNetwork()
    bob = ComplexNetwork()
    env = Env()

    state = env.reset()

    ac1 = alice(state)
    ac2 = bob(state)

    cq1, cq2 = env.step(ac1, ac2)
    qq1, qq2 = env.quantum_step(ac1, ac2)
