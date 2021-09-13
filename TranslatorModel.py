import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.__d_model = d_model
        self.__embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self, tokens):
        return self.__embedding(tokens) * math.sqrt(self.__d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length):
        super().__init__()
        self.__dropout = nn.Dropout(dropout)
        positional_embedding = torch.arange(max_length).reshape(max_length, 1) / torch.pow(
            10000, torch.arange(d_model) // 2 * 2 / d_model)
        positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, tokens_embedding):
        return self.__dropout(tokens_embedding + self.positional_embedding[:tokens_embedding.size(1), :])
