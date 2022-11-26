# py
from typing import List

# nn & rl
from torch import Tensor, tensor, rand, complex64
from torch.nn import TransformerEncoderLayer, Module, Linear, Flatten

# lib
from embedding import Embedding


class ParametrizationModule(Module):
    _linear: Linear
    _flatten: Flatten

    def __init__(self, num_actions: int):
        super(ParametrizationModule, self).__init__()
        self._linear = Linear(128*256, num_actions)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        elif len(x.shape) == 2:
            x = x.view(1, -1)
        else:
            raise AssertionError
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
            enc_layer = TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
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
