import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positional_embedding = torch.arange(max_length).reshape(max_length, 1) / torch.pow(
            10000, torch.arange(d_model) // 2 * 2 / d_model)
        positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.positional_embedding[:token_embedding.size(1), :])
