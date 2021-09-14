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
    def __init__(self, d_model, dropout, max_sequence_length):
        super().__init__()
        self.__dropout = nn.Dropout(dropout)
        positional_embedding = torch.arange(max_sequence_length).reshape(max_sequence_length, 1) / torch.pow(
            10000, torch.arange(d_model) // 2 * 2 / d_model)
        positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
        positional_embedding = positional_embedding.unsqueeze(1)
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, tokens_embedding):
        return self.__dropout(tokens_embedding + self.positional_embedding[:tokens_embedding.shape[0], :])


class TranslatorModel(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, max_sequence_length, nhead, num_encoder_layers,
                 num_decoder_layers, vocabulary_size_source, vocabulary_size_target):
        super().__init__()
        self.__token_embedding_source = TokenEmbedding(d_model, vocabulary_size_source)
        self.__token_embedding_target = TokenEmbedding(d_model, vocabulary_size_target)
        self.__positional_encoding = PositionalEncoding(d_model, dropout, max_sequence_length)
        self.__transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout)
        self.__linear = nn.Linear(d_model, vocabulary_size_target)

    def forward(self, source, target, source_mask, target_mask, source_padding_mask, target_padding_mask,
                memory_key_padding_mask):
        tokens_embedding_source = self.__positional_encoding(self.__token_embedding_source(source))
        tokens_embedding_target = self.__positional_encoding(self.__token_embedding_target(target))
        outs = self.__transformer(tokens_embedding_source, tokens_embedding_target, source_mask, target_mask, None,
                                  source_padding_mask, target_padding_mask, memory_key_padding_mask)
        return self.__linear(outs)

    def encode(self, source):
        tokens_embedding = self.__positional_encoding(self.__token_embedding_source(source))
        return self.__transformer.encoder(tokens_embedding)

    def decode(self, target, memory, target_mask):
        tokens_embedding = self.__positional_encoding(self.__token_embedding_target(target))
        return self.__transformer.decoder(tokens_embedding, memory, target_mask)
