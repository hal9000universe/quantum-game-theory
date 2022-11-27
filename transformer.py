# py
from typing import List

# nn & rl
from torch import Tensor, tensor, rand, complex64, zeros, real, no_grad, cat, int64, float32
from torch.nn import TransformerEncoderLayer, Module, Linear, Flatten
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot


class UnwantedError(Exception):

    def __init__(self, message: str):
        super(UnwantedError, self).__init__(message)


SE_PRE: int = 24
RE_PRE: int = 32
PTE_PRE: int = 8
POST: int = 128


class StateEmbedding(Module):

    def __init__(self, dims: int):
        super(StateEmbedding, self).__init__()
        self._pre_mat = zeros((SE_PRE, 1), dtype=complex64)
        self._post_mat = zeros((dims, POST), dtype=complex64)
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, -1)
        elif len(x.shape) == 1:
            x = x.view(1, -1)
        else:
            raise UnwantedError(
                "Unwanted behaviour is occurring as the input "
                "of the state embedding does not have a valid shape."
            )
        x = self._pre_mat @ x @ self._post_mat
        x = real(x)
        return x


class RewardEmbedding(Module):
    _pre_mat: Tensor
    _post_mat: Tensor

    def __init__(self, num_players: int, dims: int):
        super(RewardEmbedding, self).__init__()
        self._pre_mat = zeros((RE_PRE, dims))
        self._post_mat = zeros((num_players, POST))
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        try:
            assert len(x.shape) == 3 or len(x.shape) == 2
        except AssertionError:
            raise UnwantedError(
                "Unwanted behaviour is occurring as the input "
                "of the reward embedding does not have a valid shape."
            )
        x = self._pre_mat @ x @ self._post_mat
        return x


class PlayerTokenEmbedding(Module):

    def __init__(self, num_players):
        super(PlayerTokenEmbedding, self).__init__()
        self._pre_mat = zeros((PTE_PRE, num_players))
        self._post_mat = zeros((1, POST))
        self._init_weights()

    @no_grad()
    def _init_weights(self):
        self._pre_mat = kaiming_normal_(self._pre_mat)
        self._post_mat = kaiming_normal_(self._post_mat)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.view(x.shape[0], -1, 1)
        elif len(x.shape) == 1:
            x = x.view(-1, 1)
        else:
            raise UnwantedError(
                "Unwanted behaviour is occurring as the input "
                "of the player token embedding does not have a valid shape."
            )
        x = self._pre_mat @ x @ self._post_mat
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
        if len(embedded_state.shape) == 3:
            out: Tensor = cat((embedded_state, embedded_reward, embedded_player_token), dim=1)
        elif len(embedded_state.shape) == 2:
            out: Tensor = cat((embedded_state, embedded_reward, embedded_player_token), dim=0)
        else:
            raise UnwantedError(
                "Unwanted behaviour is occurring as the input "
                "of the embedding does not have a valid shape."
            )
        return out


class ParametrizationModule(Module):
    _linear: Linear
    _flatten: Flatten

    def __init__(self, num_actions: int):
        super(ParametrizationModule, self).__init__()
        self._linear = Linear((SE_PRE + RE_PRE + PTE_PRE) * POST, num_actions)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        elif len(x.shape) == 2:
            x = x.view(1, -1)
        else:
            raise UnwantedError(
                "Unwanted behaviour is occurring as the input "
                "of the parametrization module does not have a valid shape."
            )
        x = self._linear(x)
        if x.shape[0] == 1:
            return x.view(-1)
        return x


class Transformer(Module):
    _embedding: Embedding
    _transformer_encoder_layers: List[TransformerEncoderLayer]
    _parametrization_module: ParametrizationModule

    def __init__(self, num_players: int, num_encoder_layers: int, num_actions: int):
        super(Transformer, self).__init__()
        dims: int = pow(2, num_players)
        self._embedding = Embedding(dims=dims, num_players=num_players)
        self._transformer_encoder_layers = []
        for _ in range(0, num_encoder_layers):
            enc_layer = TransformerEncoderLayer(d_model=POST, nhead=4, batch_first=True)
            self._transformer_encoder_layers.append(enc_layer)
        self._parametrization_module = ParametrizationModule(num_actions=num_actions)

    def __call__(self, state: Tensor, rewards: Tensor, player_token: Tensor) -> Tensor:
        x: Tensor = self._embedding(state, rewards, player_token)
        for enc_layer in self._transformer_encoder_layers:
            x = enc_layer(x)
        x = self._parametrization_module(x)
        return x


if __name__ == '__main__':
    state_batch: Tensor = rand(64, 4, dtype=complex64)
    reward_batch: Tensor = rand(64, 4, 2)
    player_token_batch: Tensor = rand(64, 2)
    transformer = Transformer(2, 3, 3)
    output = transformer(state_batch, reward_batch, player_token_batch)
    print(output.shape)
