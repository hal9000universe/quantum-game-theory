# py
from typing import List
from math import pi

# nn & rl
from torch import Tensor, tensor, rand, complex64, zeros, real, no_grad, cat, relu, sigmoid
from torch.nn import TransformerEncoderLayer, Module, Linear
from torch.nn.init import kaiming_normal_


class DimensionError(Exception):

    def __init__(self, message: str):
        super(DimensionError, self).__init__(message)


SE_PRE: int = 12
RE_PRE: int = 16
PTE_PRE: int = 4
POST: int = 64


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
            raise DimensionError(
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
            raise DimensionError(
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
            raise DimensionError(
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
            raise DimensionError(
                "Unwanted behaviour is occurring as the input "
                "of the embedding does not have a valid shape."
            )
        return out


class ParametrizationModule(Module):
    _num_actions: int
    _linear1: Linear
    _linear2: Linear
    _linear3: Linear
    _layers: List[Linear]

    def __init__(self, num_actions: int):
        super(ParametrizationModule, self).__init__()
        self._num_actions = num_actions
        self._linear1 = Linear((SE_PRE + RE_PRE + PTE_PRE) * POST, RE_PRE * POST)
        self._linear2 = Linear(RE_PRE * POST, POST)
        self._linear3 = Linear(POST, num_actions)
        self._layers = [self._linear1, self._linear2, self._linear3]
        self._scaling2 = tensor([pi, pi/2], requires_grad=True)
        self._scaling3 = tensor([2*pi, pi, 2*pi], requires_grad=True)
        self._translating3 = tensor([-pi, 0., -pi], requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        elif len(x.shape) == 2:
            x = x.view(1, -1)
        else:
            raise DimensionError(
                "Unwanted behaviour is occurring as the input "
                "of the parametrization module does not have a valid shape."
            )
        # apply layers
        for layer in self._layers:
            x = relu(x)
            x = layer(x)
        x = sigmoid(x)
        if self._num_actions == 3:
            x = self._scaling3 * x + self._translating3
        elif self._num_actions == 2:
            x = self._scaling2 * x
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
