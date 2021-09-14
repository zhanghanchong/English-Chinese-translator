import torch


class MaskBuilder:
    def __init__(self, padding_index, source, target):
        self.__padding_index = padding_index
        self.__source = source
        self.__target = target

    def build_masks(self):
        source_mask = torch.zeros((self.__source.shape[0], self.__source.shape[0]))
        target_mask = torch.tril(torch.ones((self.__target.shape[0], self.__target.shape[0])))
        target_mask = target_mask.masked_fill(target_mask == 0, float('-inf')).masked_fill(target_mask == 1, 0)
        source_padding_mask = (self.__source == self.__padding_index).transpose(0, 1)
        target_padding_mask = (self.__target == self.__padding_index).transpose(0, 1)
        return source_mask, target_mask, source_padding_mask, target_padding_mask
