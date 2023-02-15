# py
from typing import List, Tuple, Optional

# nn & rl
from torch import Tensor, complex64, kron, tensor
from torch.distributions import Distribution, Uniform

# lib
from base.action_space import ActionSpace, RestrictedActionSpace
from base.quantum import QuantumSystem, Ops, Operator


class Env:
    """The Env class represents a general environment for quantum games."""
    _reward_distribution: Optional[Tensor]
    _num_players: int
    _action_space: ActionSpace
    _ops: Ops
    _uniform: Distribution
    _reward_sampler: Distribution
    _state: QuantumSystem
    _J: Operator
    _nash_eq: Optional[Tensor]

    def __init__(self, num_players: int):
        """Attributes
        self._num_players: int specifying the number of players.
        self._action_space: ActionSpace determining what strategies the players can choose from.
        self._ops: Ops object containing a variety of quantum operators.
        self._uniform: A uniform probability distribution to sample reward distributions from.
        self._state: QuantumSystem which the players can manipulate.
        self._J: entanglement operator to prepare the quantum game.
        self._nash_eq: Optional[Tensor] specifying the nash equilibrium of the game."""
        self._num_players = num_players
        self._action_space = RestrictedActionSpace()
        self._ops = Ops()
        self._uniform = Uniform(-0.2, 0.2)
        self.generate_random()
        self._state = QuantumSystem(self._num_players)
        self._J = self._ops.GeneralJ.inject(self._num_players)
        self._nash_eq = None

    @property
    def nash_eq(self) -> Optional[Tensor]:
        return self._nash_eq

    @nash_eq.setter
    def nash_eq(self, nash_eq: Optional[Tensor]):
        self._nash_eq = nash_eq

    @property
    def num_players(self) -> int:
        return self._num_players

    @num_players.setter
    def num_players(self, num_players: int):
        """Sets _num_players. _reward_distribution is set to zero
        to ensure that the reward distribution and the given number of players are compatible."""
        self._num_players = num_players
        self._reward_distribution = None

    @property
    def reward_distribution(self) -> Tensor:
        return self._reward_distribution

    @reward_distribution.setter
    def reward_distribution(self, reward_distribution: Tensor):
        """Sets _reward_distribution to a normalized version of the specified reward_distribution."""
        # ensure correct shapes
        assert len(reward_distribution.shape) == 2
        assert pow(reward_distribution.shape[1], 2) == reward_distribution.shape[0]
        # normalize
        epsilon: float = 1e-8
        self._reward_distribution = reward_distribution / max(reward_distribution.norm(), epsilon)
        self._num_players = self._reward_distribution.shape[1]

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, operator: Operator):
        """Sets _J and ensures that the given operator is compatible with the number of players."""
        assert operator.mat.shape == (pow(2, self._num_players), pow(2, self._num_players))
        self._J = operator

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: ActionSpace):
        self._action_space = action_space

    def generate_random(self):
        """Generates a completely random, possibly asymmetric reward distribution."""
        shape: Tuple[int, int] = (pow(2, self._num_players), self._num_players)
        rew_dist: Tensor = self._uniform.sample(shape)
        self.reward_distribution = rew_dist

    def generate_random_symmetric(self):
        """Generates a symmetric reward distribution."""
        assert self._num_players == 2  # only works for games with two players
        rewards: Tensor = self._uniform.sample((4,))
        rew_dist: Tensor = tensor([[rewards[0], rewards[0]], [rewards[1], rewards[2]], [rewards[2], rewards[1]],
                                   [rewards[3], rewards[3]]])
        self.reward_distribution = rew_dist

    def _observation(self) -> Tensor:
        """Returns the state of the quantum system after applying the J operator."""
        return (self._J @ QuantumSystem(num_qubits=self._num_players)).state

    def reset(self, fix_inp: bool) -> Tensor:
        """Returns an observation. If fix_inp is specified as false, noise is added to the observation.
        This can serve as an exploration mechanism."""
        if fix_inp:
            return self._observation()
        else:
            rand_inp = self._uniform.sample((4,)).type(complex64)
            return self._observation() + rand_inp

    def _create_operator(self, args: Tuple[Tensor]) -> Operator:
        """Creates one operator from all the operators chosen by the players
        which can be applied to the entangled system."""
        op: Tensor = self._action_space.operator(args[0])
        for i in range(1, len(args)):
            op = kron(op, self._action_space.operator(args[i]))
        return Operator(mat=op)

    def step(self, *args) -> List[Tensor]:
        """Play a single match of the quantum game the Env represents.
        Returns a list of q-values for the agents to learn from."""
        # prepare initial_state
        self._state = self._J @ QuantumSystem(num_qubits=self._num_players)
        # create operator which applies the local unitary actions
        op = self._create_operator(args)
        # apply operator
        self._state = op @ self._state
        # apply adjoint state preparation operator
        self._state = self._J.adjoint @ self._state
        # compute q-values
        qs: List = []
        for i in range(self._num_players):
            q_i = (self._reward_distribution[:, i] * self._state.probs).sum()
            qs.append(q_i)
        return qs
