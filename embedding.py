# nn & rl
from torch import zeros, complex64, real, no_grad, Tensor, cat, tensor, int64, float32
from torch.nn import Module
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot


class StateEmbedding(Module):

    def __init__(self, dims: int):
        super(StateEmbedding, self).__init__()
        self._pre_mat = zeros((64, 1), dtype=complex64)
        self._post_mat = zeros((dims, 256), dtype=complex64)
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._pre_mat @ x.view(1, -1) @ self._post_mat
        x = real(x)
        return x


class RewardEmbedding(Module):
    _pre_mat: Tensor
    _post_mat: Tensor

    def __init__(self, num_players: int, dims: int):
        super(RewardEmbedding, self).__init__()
        self._pre_mat = zeros((64, dims))
        self._post_mat = zeros((num_players, 256))
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._pre_mat @ x @ self._post_mat
        return x


class PlayerTokenEmbedding(Module):

    def __init__(self, num_players):
        super(PlayerTokenEmbedding, self).__init__()
        self._pre_mat = zeros((10, num_players))
        self._post_mat = zeros((1, 256))
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        x = self._pre_mat @ x.view(-1, 1) @ self._post_mat
        return x


class Embedding(Module):
    _state_embedding: StateEmbedding
    _reward_embedding: RewardEmbedding
    _player_token_embedding: PlayerTokenEmbedding

    def __init__(self, dims: int, num_players: int):
        super(Embedding, self).__init__()
        self._state_embedding = StateEmbedding(dims=dims)
        self._reward_embedding = RewardEmbedding(num_players=num_players, dims=dims)
        self._player_token_embedding = PlayerTokenEmbedding(num_players=num_players)

    def __call__(self, state: Tensor, rewards: Tensor, player_token: Tensor) -> Tensor:
        embedded_state: Tensor = self._state_embedding(state)
        embedded_reward: Tensor = self._reward_embedding(rewards)
        embedded_player_token: Tensor = self._player_token_embedding(player_token)
        out: Tensor = cat((embedded_state, embedded_reward, embedded_player_token))
        return out
