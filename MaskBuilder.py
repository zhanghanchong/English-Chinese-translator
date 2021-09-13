import torch


class MaskBuilder:
    def __init__(self, device, padding_index, source, target):
        self.__device = device
        self.__padding_index = padding_index
        self.__source = source
        self.__target = target

    def build_square_subsequent_mask(self, sequence_length):
        mask = torch.tril(torch.ones((sequence_length, sequence_length), device=self.__device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        return mask

    def build_masks(self):
        source_sequence_length = self.__source.shape[0]
        target_sequence_length = self.__target.shape[0]
        source_mask = torch.zeros((source_sequence_length, source_sequence_length), device=self.__device)
        target_mask = self.build_square_subsequent_mask(target_sequence_length)
        source_padding_mask = (self.__source == self.__padding_index).transpose(0, 1)
        target_padding_mask = (self.__target == self.__padding_index).transpose(0, 1)
        return source_mask, target_mask, source_padding_mask, target_padding_mask
